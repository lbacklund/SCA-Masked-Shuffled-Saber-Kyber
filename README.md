# Secret Key Recovery Attack on Masked and Shuffled Implementations of CRYSTALS-Kyber and Saber

This repository and its submodules contain the code and deep-learning models used in the work of the paper titled *"Secret Key Recovery Attack on Masked and Shuffled Implementations of CRYSTALS-Kyber and Saber"* published in [AIHWS23](https://aihws2023.aisylab.com/), a workshop at [ACNS23](https://sulab-sever.u-aizu.ac.jp/ACNS2023/index.html).

The repository contains two folder `saber` and `kyber`. Each contains the scripts and models used in the analysis and attack on the respective algorithm. Both use the [ECC_CCT_tool](https://github.com/lbacklund/ECC_CCT_tool) which is included as a top-level submodule. It can be pulled into a cloned repository by running `git submodule update --init --recursive`.

Authors:
 - Linus Backlund (KTH Royal Institute of Technology, Stockholm, Sweden)
 - Kalle Ngo (KTH Royal Institute of Technology, Stockholm, Sweden)
 - Joel Gärtner (KTH Royal Institute of Technology, Stockholm, Sweden)
 - Elena Dubrova (KTH Royal Institute of Technology, Stockholm, Sweden)
