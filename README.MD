<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/header.png" alt="Logo" >
  </a>

</div>


# ChurnY : Customer Churn Predictor


<!-- ABOUT THE PROJECT -->
## About The Project
__ChurnY__ is a REST API service for customer churn prediction. It can predict the likelihood of churn using customer data such as customer demographics, subscription history, and behaviors.

Using ChurnY, it is possible to get an early warning of a potential customer churn,  so we can efficiently target certain customers and develop a more focused customer retention strategies 


<p align="right">(<a href="#top">back to top</a>)</p>



### Technologies 

This service is built using the following frameworks/libraries:

* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [Gunicorn](https://gunicorn.org/)
* [Scikit-learn](https://scikit-learn.org/stable/#)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE -->
## Usage



### How to run the server


1. Build a docker image
    ```sh
    docker build -t churn .
    ```
2. Run the docker container
   ```sh
   docker run -it -rm -p 9696:9696 churn
   ```

### Sending a request
1. Please refer to the __request.py__ file. It contains an input example. Execute this script after running the server.
    ```sh
    python3 request.py
    ```
2. The response is a json file containing the churn prediction.
```sh
{'Churn Probability': 0.014, 'churn': False}
```


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ABOUT THE ML Model -->
## About The Machine Learning Model





<!-- CONTACT -->
## Contact

Ryza Rynazal [Linkedin](https://www.linkedin.com/in/ryza-rynazal/) 





<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was taught by Alexey Grigorev during  the Zoomcamp held by DataTalksClub .

* [ML Zoomcamp by DataTalksClub](https://www.youtube.com/watch?v=rowoDjPc8HU&ab_channel=DataTalksClub)


<p align="right">(<a href="#top">back to top</a>)</p>



