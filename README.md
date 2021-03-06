
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<h3 align="center">Slim timeframe momentum investing</h3>

  <p align="center">
    Momentum investing with daily portfolio updates based on statistical analysis
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![overall scope](https://user-images.githubusercontent.com/84615464/177702792-554f7105-ff95-4293-8ab5-858768a17999.png)
This work models and exploits stock's next day performance based on 5 prior days' behavior. The second order derivative is investigated, and a concave behavior is sought, as in the upper figure. Within a small range of next day performance's absolute magnitude, the second order derivative and next day's performance exhibits a linear relationship as in the lower figure. ~10 Stocks within the linear range with maximal performance parameters are chosen and covariance-minimized. Such portfolio undergoes a daily update. Please refer to `Slim timeframe momentum investing paper.pdf` for further details.

![net deviation](https://user-images.githubusercontent.com/84615464/177710131-9ef41b19-cd24-4c9a-8fce-75d37efe5e75.png)

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

Run `momentumAnalysis.py` for particular market and timeframe of interest and collect raw data (~260000 data for 5 months). Analyze the correlation between stock's past behavior and next day's perfomance using a separate tool (matlab version to be updated). Based on such analysis, set desired parameters `slopeLowThresh`, `slopeHighThresh`, `devLowThresh`, `devHighThresh` on `momentumInvesting.py` and run code.

## Results
Based on stock's behavioral data mining for 5 months between 2022/1/1 ~ 2022/6/1 for KOSPI and KOSDAQ stocks, an oustanding performance has been achieved between 2022/1/1 ~ 2022/7/1 as shown below.

![Performance](https://user-images.githubusercontent.com/84615464/177713064-6949e00b-9314-4125-bd09-16cd97c817c7.png)


<!-- ROADMAP -->
## Future work
V1.0.1
- [x] Analyze daily portfolio switching amount
- [ ] Apply methodology to more timeframes and develop a market-specific mechanism
- [ ] Analyze more features 
- [ ] Apply machine learning technique to quantifying correlation

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Any contributions or suggestions are **greatly appreciated**.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jaewoo Jeong - jeong207@kaist.ac.kr

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project has been initiated during the KAIST-UNIST AI for Finance course in 2022 spring semester, ranking 1st out of 40 teams with +11.38% for the course's quantitative investment competition which lasted for one month.

<p align="right">(<a href="#top">back to top</a>)</p>


