# Research Paper
* Name: Arsh Singh
* Semester: Spr 2026
* Topic: Random Sample Consensus (RANSAC) Algorithm

## Introduction
<!- What is the algorithm/datastructure?>
<!- What is the problem it solves?>
<!- Provide a brief history of the algorithm/datastructure. (make sure to cite sources)>

[//]: # - Provide an introduction to the rest of the paper. 

The Algorithm that I want to focus on is called Random Sample Consesus or its acronym RANSAC. The method was introduced by Martin Fischler and Robert Bolles in 1981 [1]. It is a method (the authors call it a paradigm [1]) for fitting a predtermined model to experimental data with sizeable number of outliers due to gross errors. Gross errors are defined as measurement-related errors - perhaps human causes or other random errors - that introduce large outliers in data. 

[//]: # [Example of data outliers due to gross error - the one in the paper.]

The original paper demonstrated the application of RANSAC in *location determination problem* in computer vision. The method has now been applied to a wide array of problems [2, 3, 4]. I will disuss these in the section [Applications](#application).

[//]: # [Show example of location determination - the one in the paper.]

The RANSAC paradigm is more formally stated [1] as follows.

**Given**: a model that requires a minimum of $n$ data points to instantiate its free parameters (for example, 2 for a line), and a set of data points $P$ such that the number of points in $P$ is greater than $n$; $(size(P)\ge n)$.

1. Randomly select a subset $S1$ of $n$ data points from $P$ and instantiate the model. Use the instantiated model $M1$ to determine the subset $S1*$ of points in $P$ that are within some error tolerance of $M1$. The set $S1$* is called the consensus set of $S1$.

2. If $size(S1*)$ is greater than some threshold $t$, which is a function of the estimate of the number of gross errors in $P$, use $S1*$ to compute (possibly using least squares) a new model $M1*$.

3. If $size(S1*)$ is less than $t$, randomly select a new subset $S2$ and repeat the above process.

4. If, after some predetermined number of trials, no consensus set with $t$ or more members has been found, either solve the model with the largest consensus set found, or terminate in failure.


The rest of the paper is organized as follows: 

In the next section, [Analysis of Algorithm/Datastructure](#analysis-of-algorithmdatastructure), I will present the theoretical analysis of the RANSAC algorithm tryting to fit a linear and a quadratic model. [Maybe: I will also generalize this to a k-neighbors classification problem.] I will present the time and space complexity in the case of the specified models. 

In the section [Empirical Analysis](#empirical-analysis), I will present the empirical run time of the methods I implement in Python [Maybe: and C]. I will do a comparative analysis based on the models and the three variables for RANSAC. 

In the section [Application](#application) I will take a deeper dive into the various applications of RANSAC. 

In the section [Implementation](#implementation) I will present code snippets of my final implementation [maybe C, else Python]. I willdo a walk through and present a commentary on my design choices.

In conlusion, I will present a [Summary](#summary) of my findings and lessons I learnt.


## Analysis of Algorithm/Datastructure 
Make sure to include the following:
[//]: # - Time Complexity
[//]: # - Space Complexity
[//]: # - General analysis of the algorithm/datastructure

[//]: # [Linear model]

[//]: # [Quadratic model]

[//]: # [Classification in n groups]

## Empirical Analysis
- What is the empirical analysis?
- Provide specific examples / data.


## Application
- What is the algorithm/datastructure used for?
- Provide specific examples
- Why is it useful / used in that field area?
- Make sure to provide sources for your information.


## Implementation
- What language did you use?
- What libraries did you use?
- What were the challenges you faced?
- Provide key points of the algorithm/datastructure implementation, discuss the code.
- If you found code in another language, and then implemented in your own language that is fine - but make sure to document that.


## Summary
- Provide a summary of your findings
- What did you learn?


## LLM Use Disclosure 
I used Calude to create a 4-week study plan to study this topic using the seminal paper and standard open source textbooks. 

## References

[1] Fischler, M. A. and Bolles, R. C. 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24, 6 (June 1981), 381–395. https://doi.org/10.1145/358669.358692.

[2] Szeliski, R. 2022. Computer Vision: Algorithms and Applications (2nd ed.). Springer. ISBN 978-3-030-34371-2. https://doi.org/10.1007/978-3-030-34372-9.

[3] Davies, E. R. 2012. Computer and Machine Vision: Theory, Algorithms, Practicalities (4th ed.). Academic Press. ISBN 978-0-12-386908-1.

[4] Richard Hartley and Andrew Zisserman. 2004. Multiple View Geometry in Computer Vision. Cambridge University Press.