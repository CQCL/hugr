{ pkgs, ... }:

{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  # packages = with pkgs; [ cargo rustc rust-analyzer rustfmt clippy ];

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to hugr dev shell!";

  enterShell = ''
    hello
    cargo --version
  '';

  # https://devenv.sh/languages/
  # https://devenv.sh/reference/options/#languagesrustversion
  languages.rust.enable = true;

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
