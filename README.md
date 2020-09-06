# CS 725 (Autumn 2020): Assignment 1

This assignment is due by 11.55 pm on September 20, 2020. For each additional day after Sep 20th until Sep 22nd, there will be a 10% reduction in marks. The submission portal on Moodle will close at 11.55 pm on Sep 22nd.

## Please read the following important instructions before getting started on the assignment.
1. This assignment should be completed individually.
2. This assignment is entirely programming-based and is hosted as a Kaggle competition. Please refer to Moodle Announcement for further instructions on how to access the Kaggle competition.
3. Please signup on Kaggle using your IITB LDAP email accounts, with Kaggle `Display Name` as your roll number. 


## Learning linear predictors
Your goal would be to build a linear regression model to predict the number of shares of a news article based on various features listed below

### Attribute information
	 1. timedelta:                     Days between the article publication and
	                                   the dataset acquisition
	 2. n_tokens_title:                Number of words in the title
	 3. n_tokens_content:              Number of words in the content
	 4. n_unique_tokens:               Rate of unique words in the content
	 5. n_non_stop_words:              Rate of non-stop words in the content
	 6. n_non_stop_unique_tokens:      Rate of unique non-stop words in the
	                                   content
	 7. num_hrefs:                     Number of links
	 8. num_self_hrefs:                Number of links to other articles
	                                   published by Mashable
	 9. num_imgs:                      Number of images
	10. num_videos:                    Number of videos
	11. average_token_length:          Average length of the words in the
	                                   content
	12. num_keywords:                  Number of keywords in the metadata
	13. data_channel_is_lifestyle:     Is data channel 'Lifestyle'?
	14. data_channel_is_entertainment: Is data channel 'Entertainment'?
	15. data_channel_is_bus:           Is data channel 'Business'?
	16. data_channel_is_socmed:        Is data channel 'Social Media'?
	17. data_channel_is_tech:          Is data channel 'Tech'?
	18. data_channel_is_world:         Is data channel 'World'?
	19. kw_min_min:                    Worst keyword (min. shares)
	20. kw_max_min:                    Worst keyword (max. shares)
	21. kw_avg_min:                    Worst keyword (avg. shares)
	22. kw_min_max:                    Best keyword (min. shares)
	23. kw_max_max:                    Best keyword (max. shares)
	24. kw_avg_max:                    Best keyword (avg. shares)
	25. kw_min_avg:                    Avg. keyword (min. shares)
	26. kw_max_avg:                    Avg. keyword (max. shares)
	27. kw_avg_avg:                    Avg. keyword (avg. shares)
	28. self_reference_min_shares:     Min. shares of referenced articles in
	                                   Mashable
	29. self_reference_max_shares:     Max. shares of referenced articles in
	                                   Mashable
	30. self_reference_avg_sharess:    Avg. shares of referenced articles in
	                                   Mashable
	31. weekday_is_monday:             Was the article published on a Monday?
	32. weekday_is_tuesday:            Was the article published on a Tuesday?
	33. weekday_is_wednesday:          Was the article published on a Wednesday?
	34. weekday_is_thursday:           Was the article published on a Thursday?
	35. weekday_is_friday:             Was the article published on a Friday?
	36. weekday_is_saturday:           Was the article published on a Saturday?
	37. weekday_is_sunday:             Was the article published on a Sunday?
	38. is_weekend:                    Was the article published on the weekend?
	39. LDA_00:                        Closeness to LDA topic 0
	40. LDA_01:                        Closeness to LDA topic 1
	41. LDA_02:                        Closeness to LDA topic 2
	42. LDA_03:                        Closeness to LDA topic 3
	43. LDA_04:                        Closeness to LDA topic 4
	44. global_subjectivity:           Text subjectivity
	45. global_sentiment_polarity:     Text sentiment polarity
	46. global_rate_positive_words:    Rate of positive words in the content
	47. global_rate_negative_words:    Rate of negative words in the content
	48. rate_positive_words:           Rate of positive words among non-neutral
	                                   tokens
	49. rate_negative_words:           Rate of negative words among non-neutral
	                                   tokens
	50. avg_positive_polarity:         Avg. polarity of positive words
	51. min_positive_polarity:         Min. polarity of positive words
	52. max_positive_polarity:         Max. polarity of positive words
	53. avg_negative_polarity:         Avg. polarity of negative  words
	54. min_negative_polarity:         Min. polarity of negative  words
	55. max_negative_polarity:         Max. polarity of negative  words
	56. title_subjectivity:            Title subjectivity
	57. title_sentiment_polarity:      Title polarity
	58. abs_title_subjectivity:        Absolute subjectivity level
	59. abs_title_sentiment_polarity:  Absolute polarity level
	60. target:	Number of shares

### General Instructions

1. [This](https://github.com/awasthiabhijeet/cs725-2020-assign1) repository contains the code you need to get started. For any doubts or queries, please [raise an issue](https://github.com/awasthiabhijeet/cs725-2020-assign1/issues) on the GitHub repository itself.

2. You need to implement
	- Analytical solution for L2-regularized linear regression that minimizes mean squared error function. (taught in lecture 4b)
	- An iterative solution using gradient descent for L2-regularized linear regression that minimizes mean squared error function (taught in lecture 4c)

3. [LR.py](https://github.com/awasthiabhijeet/cs725-2020-assign1/blob/master/LR.py) in the GitHub repository contains the starter code. You simply need to fill in the code inside various functions. Please carefully read the comments inside the body of each function 

4. Please do not modify argument list of functions defined in LR.py unless stated explicitly inside the comments inside the function body.

5. Feel free to transform original features using basis functions, feature scaling etc.

6. Use of any libraries apart from those used in LR.py is NOT allowed.

Tip: You will find it useful to apply some form of feature scaling to your inputs before running gradient descent on your loss function. Also tuning the hyperparameters like learning rate, regularization weight etc. should improve your results.



### How will you be graded:
1. Leaderboard Ranking: 15 points
2. Correctness of Analytical Solution: 30 points  (`analytical_solution` function in LR.py)
3. Correctness of Gradient Descent: 30 points (`compute_gradients` and `update_weights` function in LR.py)
4. A quick moodle quiz shortly after the release of a reference code: 25 points
5. 5 extra points for a properly vectorized implementation. Vectorization significantly reduces the computational run time. This will be evaluated based on the run time of functions on inputs of various sizes.
6. We will be unable to evaluate your code if you modify the argument list of `analytical_solution` or `compute_gradients` functions in LR.py. 

### Submission Instructions

1. Create a directory of name <roll_no> and copy LR.py inside it. Compress the directory as <roll_no>.tar.xz and submit to moodle before the deadline. Submissions that do not strictly adhere to this structure will be not be evaluated. 
2. Submit your predictions on Kaggle before the deadline.
3. Please do not modify argument list of functions unless mentioned in the comments inside the function body.
4. Particularly, we will be unable to evaluate your code incase you modify the argument list of `analytical_solution` and `compute_gradients` functions in LR.py. (i.e.  do not add additional arguments / remove existing arguments from these two functions.)
5. Successful submission of the assignment would include: (A) Submitting <roll_no>.tgz on Moodle and (B) Having your roll number appear on the Kaggle leaderboard.

### Help
* For any queries regarding the code/problem-statement please raise an issue on GitHub repository itself.
* Or ask your doubt on the assignment-1 thread on the MS-teams channel for the course
* Links to some tutorials
	- [tutorial_1](https://www.geeksforgeeks.org/vectorization-in-python/), [tutorial_2](https://realpython.com/numpy-array-programming/)
	- [pandas doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html), [Reading CSV files using pandas](https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/)
	- [Basic python, numpy for ML](https://cs231n.github.io/python-numpy-tutorial/)
	- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
	- Practical tips from Andrew Ng's course: [Video-1](https://youtu.be/gV5fD8Xbwgk), [Video-2](https://youtu.be/zLRB4oupj6g)
	- [Early stopping lecture](https://youtu.be/zm5cqvfKO-o) from NPTEL

