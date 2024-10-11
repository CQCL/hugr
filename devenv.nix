{ pkgs, lib, inputs, config, ... }:
let
  pkgs-stable = import inputs.nixpkgs-stable { system = pkgs.stdenv.system; };
  cfg = config.hugr;
in
{
  options.hugr = {
    setupInShell = lib.mkEnableOption "setupInShell" // {
      default = true;
      description = "run `just setup` on entering shell";
    };
  };

  config = {
    # https://devenv.sh/packages/
    # on macos frameworks have to be explicitly specified
    # otherwise a linker error ocurs on rust packages
    packages = [
      pkgs.just
      pkgs.llvmPackages_16.libllvm
      # cargo-llvm-cov is currently marked broken on nixpkgs unstable
      pkgs-stable.cargo-llvm-cov
      pkgs.graphviz
      pkgs.cargo-insta
      pkgs.capnproto
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
    '' + lib.optionalString cfg.setupInShell ''
      just setup
    '' + ''
      source .venv/bin/activate
    '';

    languages.python = {
      enable = true;
      uv = {
        enable = true;
      };
    };

    # https://devenv.sh/languages/
    # https://devenv.sh/reference/options/#languagesrustversion
    languages.rust = {
      channel = "stable";
      enable = true;
      components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
    };
  };
  # See full reference at https://devenv.sh/reference/options/
}
