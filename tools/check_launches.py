#!/usr/bin/env python3
"""Prove every kernel launch in cuda.rs passes the argument list its PTX declares.

cudarc's `launch` is unsafe for three reasons it cannot check: the argument
list may not match the kernel's signature, the kernel may write through an
argument passed as `&`, and the kernel may index out of bounds. The first is
the one that is checkable from outside the kernel, because every kernel's PTX
is in the repository -- checked in as `.ptx` or embedded as a string constant
in `cuda_kernels/mod.rs` -- and PTX declares each parameter's width.

For every `let <var> = self.kernels.get("<name>")` and every
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
kdir = root / "crates/axonml-core/src/backends/cuda_kernels"

sigs = {}
sources = list(kdir.glob("*.ptx")) + [kdir / "mod.rs"]
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
            flags.append("write" if (is_ptr and not is_const) else ("read" if is_ptr else "scalar"))
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
for f in list(kdir.glob("*.ptx")) + [kdir / "mod.rs"]:
    for m in re.finditer(r"\.entry\s+(\w+)\s*\(.*?\)\s*\{(.*?)\n\}", f.read_text(), re.S):
        if re.search(r"setp\.(ge|lt|geu|ltu)\.u32[^\n]*\n\s*@%p\d+\s+bra", m.group(2)[:1600]):
            guarded_ptx.add(m.group(1))
guarded = guarded_cu | (guarded_ptx - has_cu)

src = (root / "crates/axonml-core/src/backends/cuda.rs").read_text()

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

# var -> kernel name, per `let VAR = self.kernels.get("NAME")`
gets = [(m.start(), m.group(1), m.group(2))
        for m in re.finditer(r"let\s+(\w+)\s*=\s*self\s*\.kernels\s*\.get\(\"(\w+)\"\)", src)]

def kernel_for(var, pos):
    best = None
    for p, v, name in gets:
        if v == var and p < pos:
            best = name
    return best

problems, checked = [], 0
for m in re.finditer(r"launch_builder\((&?\*?)(\w+)\)", src):
    var = m.group(2)
    name = kernel_for(var, m.start())
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
    if name is None:
        problems.append(f"{var}: launch_builder with no preceding kernels.get"); continue
    if name not in sigs:
        problems.append(f"{name}: no .entry found in any PTX"); continue
    want = sigs[name]
    if len(args) != len(want):
        problems.append(f"{name}: rust passes {len(args)} args, ptx declares {len(want)}"); continue
    # Enclosing fn signature, for parameter mutability.
    fn_start = src.rfind("pub fn ", 0, m.start())
    fn_body = src[fn_start:m.start()]
    if name in mutable_params:
        flags = mutable_params[name]
        if len(flags) == len(args):
            for i, (a, fl) in enumerate(zip(args, flags)):
                if fl == "write":
                    got = rust_param_mutability(fn_body, a)
                    if got == "read":
                        problems.append(
                            f"{name}: param {i} is written by the kernel but Rust passes `{a.strip()}` "
                            f"as & -- cudarc records a READ event, so a later launch will not wait for this write")
    if "launch_config(" in fn_body and name not in guarded:
        problems.append(
            f"{name}: launched via launch_config(len) but no index guard found in its "
            f".cu or PTX -- the trailing threads of the last block will index past `len`")
    for i, (a, w) in enumerate(zip(args, want)):
        a = a.strip()
        is_scalar = a.startswith("&(") or a.startswith("&mut (") or re.match(r"&\w+_(u32|f32|i32|u8)\b", a) or re.match(r"&(\w+)$", a) and not any(k in a for k in ("buf", "slice", "ptr", "w", "out", "data"))
        if w == "u64" and (a.startswith("&(") or re.search(r"as (u32|f32|i32)\)", a)):
            problems.append(f"{name}: param {i} is a pointer (.u64) but rust passes scalar `{a}`")
        if w in ("u32", "f32", "s32", "u16", "u8", "b8") and not (a.startswith("&") ):
            problems.append(f"{name}: param {i} is scalar (.{w}) but rust passes `{a}`")
    checked += 1

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

print(f"kernels with a PTX signature: {len(sigs)}")
print(f"streams created in axonml-core: {len(stream_sites)} (must be 1)")
print(f"kernels with a confirmed thread-index guard: {len(guarded)}")
print(f"kernels with .cu source (write-vs-&mut checked): {len(mutable_params)}; "
      f"the rest are inline PTX and get the signature check only")
print(f"launches verified: {checked}")
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for p in problems: print("  " + p)
    sys.exit(1)
print("every launch matches its kernel's declared parameter list")
