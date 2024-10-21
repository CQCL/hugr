{ sources ? import nix/sources.nix
, pkgs ? import sources.nixpkgs {}
, lib ? pkgs.lib
, crane ? import sources.crane { inherit pkgs; }
, stdenv ? pkgs.stdenv
, llvmVersion ? "14"
, llvmPackages ? pkgs."llvmPackages_${llvmVersion}"
, libffi ? pkgs.libffi
, libxml2 ? pkgs.libxml2
}: let
  inherit (llvmPackages) libllvm;
  commonArgs = {
    src = lib.cleanSourceWith {
      src = crane.path ./.;
      filter = path: t: crane.filterCargoSources path t || (builtins.match ".*\.snap" path != null);
    };
    strictDeps = true;
    buildInputs = [
      libllvm
      libffi
      libxml2
    ];
    # TODO really we should get the version string from llvmPackages
    "LLVM_SYS_${llvmVersion}0_PREFIX" = "${libllvm.dev}";
  };
  hugr-llvm = crane.buildPackage(commonArgs // {
    cargoArtifacts = crane.buildDepsOnly commonArgs;
  });
in hugr-llvm
