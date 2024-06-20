{ pkgs, lib, config, inputs, ... }:
let
  cfg = config.hugr-llvm;
  libllvm = pkgs."llvmPackages_${cfg.llvmVersion}".libllvm.dev;

in {
  options.hugr-llvm = {
    llvmVersion = lib.mkOption {
      type = lib.types.str;
      default = "14";
    };
  };

  config = {
    packages = [
      libllvm
      pkgs.libffi
      pkgs.libxml2
    ];

    env = {
      "LLVM_SYS_${cfg.llvmVersion}0_PREFIX" = "${libllvm}";
    };

    languages.rust = {
      enable = true;
      channel = "stable";
    };

    languages.python = {
      enable = true;
      poetry = {
        enable = true;
        activate.enable = true;
      };
    };
  };
}
