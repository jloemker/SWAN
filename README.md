# CERN Summer Project 2021

The project contains a few notebooks to analyze CMS-PPS data in view of the future HL-LHC project.
The project began with the UpRoot4_example.ipyn and the Exploration-notebook.ipyn . In the Analyse_* notebooks been used to find the best dataframes and selection cuts. The final results are created with the Final_Plots.ipyn and Analysis_Error.ipyn. The StatisticTools.py includes the "statistical" treatment and the concrete dataframe construction - it uses MyHelper.py for that.

## Recommended way to run the exercise (SWAN)
[![SWAN](https://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://cern.ch/swanserver/cgi-bin/go/?projurl=https://gitlab.cern.ch/mpitt/cern-summer-project-2021.git)

To run the notebooks with regular CERN resources:
* Open a [SWAN session](https://swan.cern.ch) (the defaults are good, as of writing this pick software stack 100 and make sure to use Python3)
* In the SWAN session, click on the item on the right-hand side that says "Download Project from git" ![Download Project from git](img/download_project_trim.png)
* Copy-paste https://gitlab.cern.ch/mpitt/cern-summer-project-2021.git

## Convert ROOT files to h5py (the code is not ready for now)

Some notebooks are prepared with h5py input files. To convert ROOT files into the h5 format; we need to setup Python3 and install a few additional packages. One of the advantages of working on lxplus is a wide variety of tools and packages available for users. All packages (tools) are recompiled and provided in SFT (SoFTware Development for Experiments working group) platforms (LCG Releases). Search for the package in http://lcginfo.cern.ch/ and install the platform. In lxplus, we will use the 100 release.

```console
source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc8-opt/setup.sh
```
now you can use python3 to execute python scripts. If you are interested in installing additional python modules (like uproot4) that we are using, you can run pip with --user tag. To install needed samples:
```console
python -m pip install --user uproot4 awkward1 mplhep
```
to convert ROOT file to h5py execute:
```console
#python3 create_table.py --files filename.root --label filename
python3 create_table.py --files /eos/cms/store/user/jjhollar/CERNSummerStudentProject2021/gammagammaMuMu_FPMC_pT25_PU140_NTUPLE_1_version2.root --label gammagammaMuMu_FPMC_pT25_PU140_NTUPLE_1_version2
```

