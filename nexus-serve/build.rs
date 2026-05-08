fn main() {
    #[cfg(feature = "hailo10h")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("cargo:rustc-link-search=native={manifest_dir}");
        println!("cargo:rustc-link-search=native={manifest_dir}/hailo-aarch64/lib");
        println!("cargo:rustc-link-lib=static=hailo_genai_shim");
        println!("cargo:rustc-link-lib=static=hailo_infer_shim");

        if let Ok(dir) = std::env::var("HAILO_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        } else {
            println!("cargo:rustc-link-search=native=/usr/lib");
        }
        println!("cargo:rustc-link-lib=dylib=hailort");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
