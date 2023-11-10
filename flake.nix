{
  description = "HUGR - Hierarchical Unified Graph Representation";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs;
    nixpkgs-stable.url = github:NixOS/nixpkgs/nixos-23.05;
    devenv.url = github:cachix/devenv;
    nix2container.url = github:nlewo/nix2container;
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = github:rrbutani/nix-mk-shell-bin;
    fenix.url = github:nix-community/fenix;
    fenix.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw= nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=";
    substituters = "https://devenv.cachix.org https://nix-community.cachix.org";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        packages.default = pkgs.callPackage ./default.nix {};
        devenv.shells.default = {
          name = "hugr";
          imports = [
            ./devenv.nix
          ];
        };
      };
      flake = {};
    };
}
