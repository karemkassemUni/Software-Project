# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    
    # python 
    (python3.withPackages (ps: with ps; [
      pandas
      numpy
      setuptools
      scikit-learn
      matplotlib
    ]))
    python3.pythonOnBuildForHost  # This provides Python.h
    
    # C 
    gcc
    valgrind
    gnumake

    # test
    tmux

  ];

  shellHook = ''
    export PYTHONPATH="${pkgs.python3}/lib/python${pkgs.python3.pythonVersion}/site-packages"
    export C_INCLUDE_PATH="${pkgs.python3.pythonOnBuildForHost}/include/python${pkgs.python3.pythonVersion}"

    echo "Available packages: pandas, numpy, setuptools, scikit-learn, matplotlib"
    python --version  
    '';
}
