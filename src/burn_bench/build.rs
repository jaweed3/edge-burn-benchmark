use burn_onnx::ModelGen;
use std::path::Path;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let local_path = Path::new(&manifest).join("mobilenetv2-7.onnx");
    let repo_path = Path::new(&manifest).join("../../models/mobilenetv2-7.onnx");

    let model_path = if local_path.exists() {
        local_path.clone()
    } else if repo_path.exists() {
        std::fs::copy(&repo_path, &local_path).expect("Failed to copy model to build dir");
        local_path.clone()
    } else {
        panic!("ONNX model not found. Copy mobilenetv2-7.onnx into {}", local_path.display());
    };

    println!("cargo:rerun-if-changed={}", model_path.display());

    ModelGen::new()
        .input(model_path.to_str().unwrap())
        .out_dir("burn_model/")
        .run_from_script();
}
