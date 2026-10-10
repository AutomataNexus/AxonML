#!/usr/bin/env python3
"""Prove every kernel launch in axonml-core's cuda.rs and axonml-subbit's lib.rs passes the
argument list its PTX declares.

cudarc's `launch` is unsafe for three reasons it cannot check: the argument
list may not match the kernel's signature, the kernel may write through an
argument passed as `&`, and the kernel may index out of bounds. The first is
the one that is checkable from outside the kernel, because every kernel's PTX
is in the repository -- checked in as `.ptx` or embedded as a string constant
in `cuda_kernels/mod.rs` -- and PTX declares each parameter's width.

For every `let <var> = self.kernels.get("<name>")` (core) or `subbit_func("<name>")`
(subbit) and every
`launch_builder(<var>) .arg(..) ... .launch(..)` that uses it, this compares
the number of `.arg()` calls against the number of `.param` entries, and the
width class of each: a `.u64` parameter must receive a device pointer (a
CudaSlice / CudaView argument), a `.u32`/`.f32`/... parameter must receive a
scalar reference. A mismatch is a real defect: the kernel would read garbage
for the parameters past the mismatch.

Exit status is non-zero on any mismatch, so this can gate CI.
"""
import re, sys, glob, pathlib

root = pathlib.Path(__file__).resolve().parents[1]
LOOKUP = r"(?:self\s*\.kernels\s*\.get|subbit_func|nvfp4_func)\(\"(\w+)\"\)"
LOOKUP_CALL = r"(?:self\s*\.kernels\s*\.get|subbit_func|nvfp4_func)\("

def literals_in_lookup(text, var):
    """Every "name" literal inside `let VAR = <lookup fn>( ... )`, walking balanced
    parentheses so `subbit_func(if v4 { "a" } else { "b" })` yields both names."""
    names = []
    for m in re.finditer(r"let\s+" + re.escape(var) + r"\s*=\s*" + LOOKUP_CALL, text):
        k, depth = m.end(), 1
        while k < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[k], 0); k += 1
        names.extend(re.findall(r"\"(\w+)\"", text[m.end():k - 1]))
    return list(dict.fromkeys(names))
CRATES = [
    # (kernel dir, extra signature sources, launch source, name)
    (root / "crates/axonml-core/src/backends/cuda_kernels", ["mod.rs"], root / "crates/axonml-core/src/backends/cuda.rs", "axonml-core"),
    (root / "crates/axonml-subbit/kernels", [], root / "crates/axonml-subbit/src/lib.rs", "axonml-subbit"),
]
problems, checked, nsigs, nguarded, ncu = [], 0, 0, 0, 0
for kdir, extra, src_path, crate_name in CRATES:

    sigs = {}
    sources = list(kdir.glob("*.ptx")) + [kdir / e for e in extra]
    for f in sources:
        text = f.read_text()
        for m in re.finditer(r"\.entry\s+(\w+)\s*\((.*?)\)", text, re.S):
            widths = re.findall(r"\.param\s+\.(\w+)", m.group(2))
            sigs[m.group(1)] = widths

    # Mutability from the .cu sources: a pointer parameter without `const` is one
    # the kernel may write. That parameter must arrive from Rust as `&mut`, so
    # cudarc records a write event on the slice and later launches wait for it.
    # Passed as `&`, cudarc records a read, and a concurrent kernel can race the
    # write while the borrow checker sees two shared borrows and stays silent.
    mutable_params = {}
    for f in kdir.glob("*.cu"):
        text = f.read_text()
        for m in re.finditer(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{", text, re.S):
            params = [x.strip() for x in m.group(2).split(",") if x.strip()]
            flags = []
            for prm in params:
                is_ptr = "*" in prm
                is_const = prm.startswith("const ") or " const " in prm.split("*")[0]
                # `float* const* ptrs`: the argument is a read-only TABLE of device
                # pointers; the kernel writes through the table to buffers that are
                # not launch arguments. Ordering for those is the single stream's.
                is_table = re.search(r"\*\s*const\s*\*", prm) is not None
                flags.append("table" if is_table else "write" if (is_ptr and not is_const) else ("read" if is_ptr else "scalar"))
            mutable_params[m.group(1)] = flags

    # Index guards. A kernel launched through launch_config(len) is given exactly
    # enough threads for `len` elements, rounded up to a block, so the trailing
    # threads of the last block must exit before indexing. That guard is an
    # `if (idx < n)` in .cu, or a setp-and-bra on the thread index in PTX.
    # The .cu and the .ptx are both checked in and are not regenerated together,
    # so a guard removed from the .cu would survive in the stale .ptx. Where a
    # kernel has .cu source, that source is authoritative and must carry the guard
    # itself; PTX evidence is accepted only for kernels that exist as inline PTX
    # with no .cu at all.
    guarded_cu, has_cu, guarded_ptx = set(), set(), set()
    for f in kdir.glob("*.cu"):
        text = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", f.read_text(), flags=re.S))
        for m in re.finditer(r"__global__\s+void\s+(\w+)\s*\(.*?\)\s*\{(.*?)\n\}", text, re.S):
            has_cu.add(m.group(1))
            if re.search(r"if\s*\(\s*\w+\s*(<|>=)\s*\w+\s*\)", m.group(2)):
                guarded_cu.add(m.group(1))
    for f in list(kdir.glob("*.ptx")) + [kdir / e for e in extra]:
        for m in re.finditer(r"\.entry\s+(\w+)\s*\(.*?\)\s*\{(.*?)\n\}", f.read_text(), re.S):
            if re.search(r"setp\.(ge|lt|geu|ltu)\.u32[^\n]*\n\s*@%p\d+\s+bra", m.group(2)[:1600]):
                guarded_ptx.add(m.group(1))
    guarded = guarded_cu | (guarded_ptx - has_cu)

    src = src_path.read_text()

    def rust_param_mutability(fn_body, argname):
        """Is `argname` declared &mut in the enclosing fn's signature, or a local
        `let mut` / `&mut` expression at the call?"""
        a = argname.strip()
        if a.startswith("&mut "):
            return "write"
        if a.startswith("&"):
            return "read"
        m = re.search(r"\b" + re.escape(a) + r"\s*:\s*&(mut\s+)?", fn_body)
        if m:
            return "write" if m.group(1) else "read"
        return "unknown"

    # Every kernel literal fetched in the enclosing fn must be compatible with the
    # launch: a fn that picks one of several (`match name { "a" => get("a"), .. }`)
    # is checked against each. A fn that compiles its own kernel with nvrtc must
    # assemble the entry text itself (`__global__ void NAME(` literal in the fn) —
    # that literal is then the signature.
    def kernels_in(fn_body):
        names = re.findall(LOOKUP, fn_body)
        return list(dict.fromkeys(names))

    def inline_entry(fn_body):
        m = re.search(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{\{", fn_body, re.S)
        if not m:
            return None
        raw = re.sub(r"\\\s*\n\s*", " ", m.group(2))
        params = [x.strip() for x in raw.split(",") if x.strip()]
        flags = []
        for prm in params:
            is_ptr = "*" in prm
            is_const = prm.startswith("const ") or " const " in prm.split("*")[0]
            flags.append("write" if (is_ptr and not is_const) else ("read" if is_ptr else "scalar"))
        guarded_inline = re.search(r"if\s*\(\s*\w+\s*(<|>=)\s*\w+\s*\)", fn_body) is not None
        return m.group(1), flags, guarded_inline

    for m in re.finditer(r"launch_builder\((&?\*?)(\w+)\)", src):
        var = m.group(2)
        fn_start = max(src.rfind("pub fn ", 0, m.start()), src.rfind("    fn ", 0, m.start()))
        fn_body_all = src[fn_start:m.start()]
        # `let VAR = self.kernels.get("NAME")` binds the launch to one kernel; a
        # VAR bound any other way (a `match` over names) is checked against every
        # literal the fn fetches.
        # The LAST binding of VAR before the launch decides; a lookup whose argument is an
        # `if`/`match` over literals is checked against each literal it can pick.
        bound = None
        for bm in re.finditer(r"let\s+" + re.escape(var) + r"\s*=\s*" + LOOKUP_CALL, fn_body_all):
            bound = bm
        candidates = literals_in_lookup(fn_body_all[bound.start():], var) if bound else kernels_in(fn_body_all)
        jit = inline_entry(fn_body_all) if not candidates else None
        end = src.find(".launch(", m.end())
        chain = src[m.end():end]
        # Each .arg(EXPR) up to the .launch(; EXPR may itself contain balanced
        # parentheses (`&(n as u32)`), so match them by walking, not by regex.
        args = []
        i = 0
        while True:
            j = chain.find(".arg(", i)
            if j < 0:
                break
            k = j + len(".arg(")
            depth = 1
            while k < len(chain) and depth:
                if chain[k] == "(":
                    depth += 1
                elif chain[k] == ")":
                    depth -= 1
                k += 1
            args.append(chain[j + len(".arg("):k - 1])
            i = k
        if not candidates and jit is None:
            problems.append(f"{crate_name} {var}: launch_builder with no kernels.get literal and no inline __global__ entry in its fn"); continue
        fn_body = fn_body_all
        if jit is not None:
            jname, jflags, jguard = jit
            if len(args) != len(jflags):
                problems.append(f"{jname} (nvrtc, inline): rust passes {len(args)} args, entry declares {len(jflags)}"); continue
            if not jguard:
                problems.append(f"{jname} (nvrtc, inline): entry text has no index guard"); continue
            for i_, (a, fl) in enumerate(zip(args, jflags)):
                if fl == "write" and rust_param_mutability(fn_body, a) == "read":
                    problems.append(f"{jname} (nvrtc, inline): param {i_} is written by the kernel but Rust passes `{a.strip()}` as &")
            checked += 1
            continue
        for name in candidates:
            if name not in sigs:
                problems.append(f"{crate_name} {name}: no .entry found in any PTX"); continue
            want = sigs[name]
            if len(args) != len(want):
                problems.append(f"{crate_name} {name}: rust passes {len(args)} args, ptx declares {len(want)}"); continue
            if name in mutable_params:
                flags = mutable_params[name]
                if len(flags) == len(args):
                    for i, (a, fl) in enumerate(zip(args, flags)):
                        if fl == "write":
                            got = rust_param_mutability(fn_body, a)
                            if got == "read":
                                problems.append(
                                    f"{crate_name} {name}: param {i} is written by the kernel but Rust passes `{a.strip()}` "
                                    f"as & -- cudarc records a READ event, so a later launch will not wait for this write")
            if "launch_config(" in fn_body and name not in guarded:
                problems.append(
                    f"{crate_name} {name}: launched via launch_config(len) but no index guard found in its "
                    f".cu or PTX -- the trailing threads of the last block will index past `len`")
            for i, (a, w) in enumerate(zip(args, want)):
                a = a.strip()
                # A .u64 parameter is a device pointer unless the .cu source declares it as a
                # 64-bit scalar (`long long n`), which PTX also spells .u64.
                cu_scalar = name in mutable_params and i < len(mutable_params[name]) and mutable_params[name][i] == "scalar"
                if w == "u64" and not cu_scalar and (a.startswith("&(") or re.search(r"as (u32|f32|i32)\)", a)):
                    problems.append(f"{crate_name} {name}: param {i} is a pointer (.u64) but rust passes scalar `{a}`")
                if w in ("u32", "f32", "s32", "u16", "u8", "b8") and not (a.startswith("&") ):
                    problems.append(f"{crate_name} {name}: param {i} is scalar (.{w}) but rust passes `{a}`")
        checked += 1
    nsigs += len(sigs); nguarded += len(guarded); ncu += len(mutable_params)
    print(f"{crate_name}: {len(sigs)} PTX signatures, {len(guarded)} index-guarded, {len(mutable_params)} with .cu source")

# The backend disables cudarc's per-slice event tracking, which is sound only
# because every operation is issued on ONE stream and CUDA serialises a
# stream's work in order. Every launch comment in cuda.rs rests on that. A
# second stream would silently void all of them, so count the streams.
core = root / "crates/axonml-core/src"
stream_sites = []
for f in core.rglob("*.rs"):
    for i, line in enumerate(f.read_text().split("\n"), 1):
        if "new_stream(" in line and not line.strip().startswith("//"):
            stream_sites.append(f"{f.relative_to(root)}:{i}")
if len(stream_sites) != 1:
    problems.append(
        f"expected exactly one stream creation in axonml-core (event tracking is off, "
        f"so ordering relies on a single stream); found {len(stream_sites)}: {stream_sites}")

print(f"kernels with a PTX signature: {nsigs}")
print(f"streams created in axonml-core: {len(stream_sites)} (must be 1)")
print(f"kernels with a confirmed thread-index guard: {nguarded}")
print(f"kernels with .cu source (write-vs-&mut checked): {ncu}; "
      f"the rest are inline PTX and get the signature check only")
print(f"launches verified: {checked}")
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for p in problems: print("  " + p)
    sys.exit(1)
print("every launch matches its kernel's declared parameter list")
