{ pkgs, lib, inputs, config, ... }:
let
  pkgs-stable = import inputs.nixpkgs-2505 { system = pkgs.stdenv.system; };
  cfg = config.hugr;
in
{
  options.hugr = {
    llvmVersion = lib.mkOption {
      type = lib.types.str;
      default = "14";
    };
  };

  config = {
    # https://devenv.sh/packages/
    # on macos frameworks have to be explicitly specified
    # otherwise a linker error occurs on rust packages
    packages = [
      pkgs.just
      pkgs.graphviz
      pkgs.cargo-insta
      pkgs.cargo-nextest
      pkgs.cargo-dist
      pkgs.capnproto

      # These are required for hugr-llvm to be able to link to llvm.
      pkgs.libffi
      pkgs.libxml2
      pkgs.ncurses
    ];

    env = {
      "LLVM_SYS_${cfg.llvmVersion}0_PREFIX" = "${pkgs-stable."llvmPackages_${cfg.llvmVersion}".libllvm.dev}";
    };


    enterShell = ''
      cargo --version
    '';

    languages.python = {
      enable = true;
      uv = {
        enable = true;
        sync.enable = true;
      };
      venv.enable = true;
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
