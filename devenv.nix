{ pkgs, lib, inputs, ... }:
let
  pkgs-stable = import inputs.nixpkgs-stable { system = pkgs.stdenv.system; };
in
{
  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified
  # otherwise a linker error ocurs on rust packages
  packages = [
    pkgs.just
    pkgs.llvmPackages_16.libllvm
    # cargo-llvm-cov is currently marked broken on nixpkgs unstable
    pkgs-stable.cargo-llvm-cov
  ] ++ lib.optionals
    pkgs.stdenv.isDarwin
    (with pkgs.darwin.apple_sdk; [
      frameworks.CoreServices
      frameworks.CoreFoundation
      # added for json schema validation tests
      frameworks.SystemConfiguration
    ]);

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to hugr dev shell!";

  enterShell = ''
    hello
    cargo --version
    export LLVM_COV="${pkgs.llvmPackages_16.libllvm}/bin/llvm-cov"
    export LLVM_PROFDATA="${pkgs.llvmPackages_16.libllvm}/bin/llvm-profdata"

    just setup
  '';

  languages.python = {
    enable = true;
    poetry = {
      enable = true;
      activate.enable = true;
    };
  };

  # https://devenv.sh/languages/
  # https://devenv.sh/reference/options/#languagesrustversion
  languages.rust = {
    channel = "stable";
    enable = true;
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  # See full reference at https://devenv.sh/reference/options/
}
