use std::path::Path;

fn main() {
    // Only run this build script on Linux systems where CUDA is expected.
    if cfg!(target_os = "linux") && std::env::var("CARGO_FEATURE_NVIDIA").is_ok() {
        println!("cargo:rerun-if-changed=src/c/cuda_reset.c");

        let cuda_path_str =
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let cuda_path = Path::new(&cuda_path_str);
        let cuda_include_path = cuda_path.join("include");
        let cuda_lib_path = cuda_path.join("lib64");

        if !cuda_include_path.exists() {
            panic!(
                "CUDA include directory not found at '{}'. Please ensure CUDA is installed correctly and the CUDA_HOME environment variable is set.",
                cuda_include_path.display()
            );
        }

        if !cuda_lib_path.exists() {
            panic!(
                "CUDA library directory not found at '{}'. Please ensure CUDA is installed correctly and the CUDA_HOME environment variable is set.",
                cuda_lib_path.display()
            );
        }

        cc::Build::new()
            .file("src/c/cuda_reset.c")
            .include(&cuda_include_path)
            .compile("cuda_reset");

        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
