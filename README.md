



<!-- ABOUT THE PROJECT -->
## HAND-FIM assessment toolset

### New version is avaialble at [https://github.com/USU-CIROH/HAND-FIM_Assessment_public](https://github.com/USU-CIROH/HAND-FIM_Assessment_public)

------------------------------------------------------------------------------------------------------------------------------------------------

### Motivation

We developed HAND-FIM assessment toolset to (1) calculate hydraulic geometry (HG) parameters such as width (W), perimeter (P), flow area (A), and hydraulic radius (R) from topographies of interest using the Height Above Nearest Drainage (HAND) method, (2) extract HG parameters from 2D hydrodynamic model with 1-m high-resolution topobathy (Benchmark), (3) compare the HAND HG parameters with their actual distribution obtained from the benchmark dataset, and (4) compare the HAND-based rating curve (e.g., synthetic rating curve) with the family of rating curves obtained from the benchmark dataset.


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* arcpy
* numpy
* pandas
* simpledbf
* matplotlib
* seaborn


<!-- USAGE EXAMPLES -->
## Usage

This toolset is composed of three main python scripts and the first script should be run before the second and thrid scripts.

#### 1. SFE_Mainstem_HAND_params_calc.py
This python script calculates the HG parameters including W, P, A, and R using the HAND method. For this example, we calculate them for 1-m DEM with bathymetry and 10-m DEM without bathymetry.
- Input
    - Topography of interest in raster format
        - 1-m DEM with bathymetry
        - 10-m DEM without bathymetry
    - HAND raster for each topography
    - Boundary polygon (.shp)
- Output
    - A table of HG parameters for different stages for each topography

#### 2. SFE_Mainstem_HAND_params_dist.py
This python script calculates the distribution of HG parameters from the benchmark dataset and plot them with the ones calculated by the HAND method.
- Input
    - Outputs from the first script (1.)
    - Topography used to run the 2D hydrodynamic model in raster format
        - 1-m DEM with bathymetry
    - Cross-section lines (.shp)
    - Thalweg points for each cross-section lines (.shp)
    - For each flow condition,
        - Water surface elevation (WSE) in raster format
        - Downstream WSE and flow discharge 
- Output
    - A plot of the distribution of HG parameters with the ones calculated by the HAND method

<p align="center" width="100%">
<img width="50%" src="/codes/SFE_Leggett_hand_param_calc/HAND_BM/6_XS_Area_log.png" alt="output2">
</p>


#### 3. SFE_Mainstem_HAND_rating_curves.py
This python script generates the HAND-based rating curve (e.g., synthetic rating curve) a the family of rating curves obtained from the benchmark dataset.
- Input
    - Outputs from the first script (1.) 
    - Topography used to run the 2D hydrodynamic model in raster format
        - 1-m DEM with bathymetry
    - Manning's n and channel slope 
    - Cross-section lines (.shp)
    - Thalweg points for each cross-section lines (.shp)
    - For each flow condition,
        - Water surface elevation (WSE) in raster format
        - Downstream WSE and flow discharge 
- Output
    - A plot of the HAND-based rating curve and the family of rating curves obtained from the benchmark dataset

<!-- ![output3](/codes/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png) -->

<p align="center" width="100%">
<img width="50%" src="/codes/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png" alt="output3">
</p>

<!-- ROADMAP 

## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish
-->


<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
-->

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Anzy Lee anzy.lee@usu.edu

Project Link: [https://github.com/anzylee/HAND-FIM_Assessment_public](https://github.com/anzylee/HAND-FIM_Assessment_public)


<!-- ACKNOWLEDGMENTS 
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

