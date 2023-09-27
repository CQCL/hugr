{ pkgs, lib, config, ... }:

{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified 
  # otherwise a linker error ocurs on rust packages
  packages = lib.optionals pkgs.stdenv.isDarwin (with pkgs.darwin.apple_sdk; [
    frameworks.CoreServices
    frameworks.CoreFoundation
    pkgs.just
  ]);

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to hugr dev shell!";

  enterShell = ''
    hello
    cargo --version
  '';

  # https://devenv.sh/languages/
  # https://devenv.sh/reference/options/#languagesrustversion
  languages.rust = {
    channel = "stable";
    enable = true;
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks.clippy.enable = true;
  pre-commit.tools.clippy = lib.mkForce config.languages.rust.toolchain.clippy;
  pre-commit.hooks.rustfmt.enable = true;
  pre-commit.tools.rustfmt = lib.mkForce config.languages.rust.toolchain.rustfmt;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
