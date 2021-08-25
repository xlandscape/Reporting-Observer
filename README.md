## Table of Contents
* [About the project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


## About the project
The `ReportingObserver` observes aquatic Landscape Model simulation runs and outputs a standard set of reports once the
experiment is finished.

### Built with
* Landscape Model core v1.5.9
* Create reporting aqRisk@LandcapeModel v0.2 (see `module\info.xml`)


## Getting Started
The following section gives instructions on how to make use of the `ReportingObserver` within your model variant. This
information is intended for model developers.

### Prerequisites
Make sure you are using the most recent version of the Landscape Model core that is compatible with this 
`ReportingObserver` version. The description here assumes a setup as described in the Landscape Model core `README`.

### Installation
1. Copy the observer to the `model\variant` folder of the Landscape Model.
2. Add the observer to the `<Parts>` section of the `model\variant\experiment.xml` file.
3. Place an `Observer` block within the `model\variant\experiment.xml` that configures the observer to be run after an
   experiment has finished. See [usage](#usage) for further details.


## Usage
The following XML snippet gives an example of a `ReportingObserver` configuration placed in the `experiment.xml`. Below,
the individual configuration options are described in more detail.

```xml
<Observer module="ReportingObserver" class="ReportingObserver">
    <Data>$(_EXP_BASE_DIR_)\$(SimID)</Data>
    <Output_Folder>$(_EXP_BASE_DIR_)\$(SimID)\reporting\$(Species1)_SD</Output_Folder>
    <Key>$(SimID)</Key>
    <CmfCont>$(RunCmfContinuous)</CmfCont>
    <Cascade>$(RunCascadeToxswa)</Cascade>
    <Steps>$(RunStepsRiverNetwork)</Steps>
    <Lguts>$(RunLGuts)</Lguts>
    <T0>$(SimulationStart)T00:00</T0>
    <Tn>$(SimulationEnd)T23:00</Tn>
    <Cmf_Depth>cmf_depth</Cmf_Depth>
    <CmfCont_PecSW>CmfCont_PecSW</CmfCont_PecSW>
    <CmfCont_Survival>CmfCont_Survival</CmfCont_Survival>
    <Steps_PecSW>Steps_PecSW</Steps_PecSW>
    <Steps_Survival>steps_survival</Steps_Survival>
    <Cascade_PecSW>Cascade_PecSW</Cascade_PecSW>
    <Cascade_Survival>cascade_survival</Cascade_Survival>
    <Reaches>$(ReportingReaches)</Reaches>
    <Pec_P1>20</Pec_P1>
    <Pec_P2>80</Pec_P2>
    <Pec_X>Reaches</Pec_X>
    <Pec_Y>PEC$_{SW}$ [$\mu$g L$^{-1}$]</Pec_Y>
    <Pec_YLim>(0,2)</Pec_YLim>
    <Pec_Ylim_Small>(0,0.5)</Pec_Ylim_Small>
    <Pec_C>b</Pec_C>
    <Pec_L>PEC$_{SW,max}$</Pec_L>
    <Pec_Func>np.max</Pec_Func>
    <Lguts_P1>20</Lguts_P1>
    <Lguts_P2>80</Lguts_P2>
    <Lguts_YLim>(0,1.1)</Lguts_YLim>
    <Lguts_X>Reaches</Lguts_X>
    <Lguts_Y>Survival rate [0-1; 1=alive]</Lguts_Y>
    <Lguts_C>g</Lguts_C>
    <Lguts_L>Survival rate</Lguts_L>
    <Lguts_Func>np.min</Lguts_Func>
    <LM_Spray_Drift_DS>/DepositionToReach/Deposition</LM_Spray_Drift_DS>
    <LM_Spray_Drift_Reaches_DS>/DepositionToReach/Reaches</LM_Spray_Drift_Reaches_DS>
    <LM_Compound_Name>CMP_A</LM_Compound_Name>
    <LM_Simulation_Start>$(SimulationStart)</LM_Simulation_Start>
    <LM_Cascade_DS>/CascadeToxswa/ConLiqWatTgtAvg</LM_Cascade_DS>
    <LM_Cmf_Continuous_DS>/CmfContinuous/PEC_SW</LM_Cmf_Continuous_DS>
    <LM_Steps_River_Network_DS>/StepsRiverNetwork/PEC_SW</LM_Steps_River_Network_DS>
    <LM_Hydrography_DS>/LandscapeScenario/hydrography</LM_Hydrography_DS>
    <LM_Cascade_Reaches>/CascadeToxswa/Reaches</LM_Cascade_Reaches>
    <LM_Cmf_Continuous_Reaches>/CmfContinuous/Reaches</LM_Cmf_Continuous_Reaches>
    <LM_Steps_River_Network_reaches>/StepsRiverNetwork/Reaches</LM_Steps_River_Network_reaches>
    <LM_Catchment>$(:Catchment)</LM_Catchment>
    <LM_Cmf_Depth_DS>/Hydrology/Depth</LM_Cmf_Depth_DS>
    <LM_Cmf_Reaches_DS>/Hydrology/Reaches</LM_Cmf_Reaches_DS>
    <LM_Steps_Survival>/IndEffect_StepsRiverNetwork_SD_Species1/GutsSurvivalReaches</LM_Steps_Survival>
    <LM_Steps_Survival_Mfs_Index>10</LM_Steps_Survival_Mfs_Index>
    <LM_Cmf_Continuous_Survival>
        /IndEffect_CmfContinuous_SD_Species1/GutsSurvivalReaches
    </LM_Cmf_Continuous_Survival>
    <LM_Cmf_Continuous_Survival_Mfs_Index>10</LM_Cmf_Continuous_Survival_Mfs_Index>
    <LM_Cascade_Survival>/IndEffect_CascadeToxswa_SD_Species1/GutsSurvivalReaches</LM_Cascade_Survival>
    <LM_Cascade_Survival_Mfs_Index>10</LM_Cascade_Survival_Mfs_Index>
    <LM_Cmf_Survival_Offset>true</LM_Cmf_Survival_Offset>
    <LM_Steps_Survival_Offset>true</LM_Steps_Survival_Offset>
    <LM_Cascade_Survival_Offset>true</LM_Cascade_Survival_Offset>
</Observer>
```

### Data
The absolute path to the experiment. This should normally equal the path given under `General\ExperimentDir` in the 
experiment configuration.

### Output_Folder
The absolute path where the final reporting elements of the `ReportingObserver` are written to. This is also the path 
used for temporary files.

### Key
The name of the experiment. This is normally specified in the user parameterization and the according parameter should
be referenced here.
    
### CmfCont
`true` or `false`. Specifies whether results of CmfContinuous simulations are used for reporting. 

### Cascade
`true` or `false`. Specifies whether results of CascadeToxswa simulations are used for reporting.
    
### Steps
`true` or `false`. Specifies whether results of StepsRiverNetwork simulations are used for reporting.

### LGuts
`true` or `false`. Specifies whether results of LEffectModel simulations are used for reporting.

### T0
The first time step that is analyzed by the reporting observer. The time stamp has to be in the format
`yyyy-mm-ddTHH:MM`.

### Tn
The last time step that is analyzed by the reporting observer. The time stamp has to be in the format
`yyyy-mm-ddTHH:MM`.

### Cmf-Depth
The `ReportingObserver` internal name that will be used for the dataset containing reach water depths. 

### CmfCont_PecSW
The `ReportingObserver` internal name that will be used for the dataset containing surface water concentrations reported
by CmfContinuous.

### CmfCont_Survival
The `ReportingObserver` internal name that will be used for the dataset containing percentages of survival as reported
by the LEffectModel on basis of concentrations simulated using CmfContinuous.

### Steps_PecSW
The `ReportingObserver` internal name that will be used for the dataset containing surface water concentrations reported
by StepsRiverNetwork.

### Steps_Survival
The `ReportingObserver` internal name that will be used for the dataset containing fractions of survival as reported by
the LEffectModel on basis of concentrations simulated using StepsRiverNetwork.

### Cascade_PecSW
The `ReportingObserver` internal name that will be used for the dataset containing surface water concentrations reported
by CascadeToxswa.

### Cascade_Survival
The `ReportingObserver` internal name that will be used for the dataset containing percentages of survival as reported
by the LEffectModel on basis of concentrations simulated using CascadeToxswa.

### Reaches
A list of numerical identifiers that specify those reaches in the scenario for which the reporting will be conducted.

### Pec_P1
The lower percentile that is plotted for concentration values over Monte Carlo runs.

### Pec_P2
The upper percentile that is plotted for concentration values over Monte Carlo runs.

### Pec_X
The x-axis label for the concentration plots. 

### Pec_Y
The y-axis label for the concentration plots.

### Pec_YLim
The range of values spanned by the y-axis of concentration plots.

### Pec_YLim_Small
Not used.

### Pec_c
The matplotlib color code used for lines showing concentrations.
    
### Pec_L
The title of the concentration plot.
    
### Pec_Func
The function that is used for the first aggregation step of concentrations.

### LGuts_P1
The lower percentile that is plotted for survival rates over Monte Carlo runs.

### LGuts_P2
The upper percentile that is plotted for survival rates over Monte Carlo runs.

### LGuts_YLim
The range of values spanned by the y-axis of survival rate plots.

### LGuts_X
The x-axis label for the survival rate plots. 

### LGuts_Y
The y-axis label for the survival rate plots.

### LGuts_C
The matplotlib color code used for lines showing survival rates.

### LGuts_L
The title of the survival rate plot.
 
### LGuts_Func
The function that is used for the first aggregation step of survival rates.

### LM_Spray_Drift_DS
The name of the Landscape Model datasets containing spray-drift depositions.

### LM_Spray_Drift_Reaches_DS
The name of the Landscape Model datasets containing reach identifiers for spray-drift depositions.

### LM_Compound_Name
The name of the substance analyzed.

### LM_Simulation_Start
The first day of the simulation. Should be the same as [T0](#t0) but without the time part.

### LM_Cascade_DS
The name of the Landscape Model datasets containing concentrations simulated by CascadeToxswa. 

### LM_Cmf_Continuous_DS
The name of the Landscape Model datasets containing concentrations simulated by CmfContinuous.

### LM_Steps_River_Network_DS
The name of the Landscape Model datasets containing concentrations simulated by StepsRiverNetwork.

### LM_Hydrography_DS
The name of the Landscape Model datasets containing the geometries of the reaches in the scenario.
    
### LM_Cascade_Reaches
The name of the Landscape Model datasets containing reach identifiers for concentrations simulated by CascadeToxswa.

### LM_Cmf_Continuous_Reaches
The name of the Landscape Model datasets containing reach identifiers for concentrations simulated by CmfContinuous.

### LM_Steps_River_Network_reaches
The name of the Landscape Model datasets containing reach identifiers for concentrations simulated by StepsRiverNetwork.

### LM_Catchment
The absolute path to the catchment description CSV file.

### LM_Cmf_Depth_DS
The name of the Landscape Model datasets containing water depths.

### LM_Cmf_Reaches_DS
The name of the Landscape Model datasets containing reach identifiers for water depths.

### LM_Steps_Survival
The name of the Landscape Model datasets containing survival rates based on concentrations simulated with
StepsRiverNetwork.

### LM_Steps_Survival_Mfs_Index
The index of the multiplication factor of survival rates based on concentrations simulated with StepsRiverNetwork that
is used to retrieve the data to plot.
    
### LM_Cmf_Continuous_Survival
The name of the Landscape Model datasets containing survival rates based on concentrations simulated with
CmfContinuous.

### LM_Cmf_Continuous_Survival_Mfs_Index
The index of the multiplication factor of survival rates based on concentrations simulated with CmfContinuous that is
used to retrieve the data to plot.

### LM_Cascade_Survival
The name of the Landscape Model datasets containing survival rates based on concentrations simulated with
CascadeToxswa.

### LM_Cascade_Survival_Mfs_Index
The index of the multiplication factor of survival rates based on concentrations simulated with CascadeToxswa that is
used to retrieve the data to plot.

### LM_Cmf_Survival_Offset
`true` or `false`. Specifies whether the survival rates are temporally offset relative to the concentrations of 
CmfContinuous.

### LM_Steps_Survival_Offset
`true` or `false`. Specifies whether the survival rates are temporally offset relative to the concentrations of 
StepsRiverNetwork.

### LM_Cascade_Survival_Offset
`true` or `false`. Specifies whether the survival rates are temporally offset relative to the concentrations of 
CascadeToxswa.


## Roadmap
The `ReportingObserver` is considered stable with no further development planned. We suggest to use `ReportingElements`,
within compositions or within Jupyter notebooks instead.


## Contributing
Contributions are welcome. Please contact the authors (see [Contact](#contact)).


## License
Distributed under the CC0 License. See `LICENSE` for more information.


## Contact
Sebastian Multsch - smultsch@knoell.com
Thorsten Schad - thorsten.schad@bayer.com
Sascha Bub - sascha.bub@gmx.de


## Acknowledgements
* [GDAL](https://pypi.org/project/GDAL)
* [h5py](https://www.h5py.org)
* [NumPy](https://numpy.org)
