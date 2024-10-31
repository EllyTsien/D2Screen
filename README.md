# DeepScreenZine


## How to start

## Preparation for input data

 1. fientne dataset (input.csv) in datasets/input.csv floder
 ```
 # formate example
ID,SMILES,label
116363,c1ccc(-c2ccccn2)nc1,0
103573,COc1ccc(-c2cc(=O)c3c(OC)c(OC)c(OC)c(OC)c3o2)cc1,0
104712,N#Cc1cnc2cnc(NCc3cccnc3)cc2c1Nc1ccc(F)c(Cl)c1,0
110269,COc1ccc2c(c1)CN(C(=O)CCl)C(c1ccccc1)C2,1
```

2. receptor.pdbqt
Preprocessed receptor file in the pdbqt formate under datasets/receptor.pdbqt. Require to remove solvent and crystal ligand. 
There is a website where you can easily transfer you .pdb file to .pdbqt. 
```
https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html
```

3. native_lig.pdb
co-crystal ligand structure (pdb format) from the target protein should be put under datasets/target_protein/native_lig.pdb
This is used for calculation of the grid center and boxsize.


## step 1: finetune on downstream task
```
python train.py
```

## step 2: transfer smile to pdbqt
We provide ligand_prep.py for fast convertion of smiles representation of molecules to pdbqt format, which can be put into autodock vina. For conformation optimization, MMFF94 is ussed. All hytrogen is kept. 
```
python smile2pdbqt.py
```
We highly recommend you to upload your own prepared pdbqt file for more accurate 3D comformation. To do this, upload your pdbqt files under the datasets/ligand_prep/ floder. 


## step 3: docking by vina
grid center and boxsize is calculated by LaBox algrithm
Cite: Ryan Loke. (2023). RyanZR/LaBOX: LaBOX v1.0.2 (v1.0.2). Zenodo. https://doi.org/10.5281/zenodo.8241444
```
python docking.py
```

## reference
@article{fang2022geometry,
  title={Geometry-enhanced molecular representation learning for property prediction},
  author={Fang, Xiaomin and Liu, Lihang and Lei, Jieqiong and He, Donglong and Zhang, Shanzhuo and Zhou, Jingbo and Wang, Fan and Wu, Hua and Wang, Haifeng},
  journal={Nature Machine Intelligence},
  pages={1--8},
  year={2022},
  publisher={Nature Publishing Group},
  doi={10.1038/s42256-021-00438-4}
}

<div id="refer-anchor-1"></div>

- [1] [百度学术](http://xueshu.baidu.com/)

<div id="refer-anchor-2"></div>

- [2] [Wikipedia](https://en.wikipedia.org/wiki/Main_Page)
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/u012349679/article/details/103815049