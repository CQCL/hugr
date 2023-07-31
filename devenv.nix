{ pkgs, lib, ... }:

{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  # on macos frameworks have to be explicitly specified 
  # otherwise a linker error ocurs on rust packages
  packages = lib.optionals pkgs.stdenv.isDarwin (with pkgs.darwin.apple_sdk; [
    frameworks.CoreServices
    frameworks.CoreFoundation
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
    enable = true;
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks.clippy.enable = true;
  pre-commit.hooks.rustfmt.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
