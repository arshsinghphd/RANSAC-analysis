# RANSAC - Random Sample Consensus

## Project Idea - Implement and Analyze RANSAC Algorithm

Many real-world computer vision tasks require a field of view far wider than what a single camera can capture. Image stitching addresses this by combining multiple overlapping images into a single seamless panorama or mosaic. This is valuable in applications such as satellite and aerial imaging, medical imaging, autonomous navigation, and augmented reality — anywhere a broader spatial context is needed than a single frame can provide. Beyond panoramas (homography), stitching is also foundational to constructing large-scale scene representations and maps from collections of images taken from different viewpoints (Structure from Motion or 3-D reconstruction).

In 2-D image stitching, the goal is to align two or more overlapping images by estimating a geometric transformation — such as a homography — that maps points from one image to corresponding points in another. This requires finding reliable feature correspondences between images. However, automated feature matching is inherently noisy: many matched point pairs will be incorrect, either due to repetitive textures, illumination differences, or the limitations of the feature detector itself. These incorrect matches, or outliers, can severely distort the estimated transformation if not handled properly. A robust fitting method is therefore essential to recover an accurate homography from data that may be majority outliers.

RANSAC (Random Sample Consensus) [1] is one of the most widely used tools for outlier rejection and data fitting, particularly in 2-D image stitching and structure from motion. It works by repeatedly attempting to identify a set of inliers from the data until the quality of fit surpasses a given criterion. As the name implies, it is a randomized algorithm — more specifically, a randomized iterative matching process.

As background, the number of data points `n` needed to specify a potential fit equals the number of degrees of freedom of the data: two for a straight line in a plane, three for a circle, four for a sphere, and so on. The remaining parameter to specify is the number of iterations `k` over sets of `n` data points required to arrive at the final best-fit solution.

The RANSAC technique proceeds as follows:

1. **Randomly Select** `k` correspondence sets as hypothetical solutions.
2. **Compute** an initial residual estimate for each set.
3. **Define** a threshold residual distance `t` to classify inliers. This threshold need not be predetermined — it can be derived from the data itself, for example by setting `t` equal to three standard deviations.
4. **Evaluate** consensus: for each hypothetical solution, count the number of points that qualify as inliers.
5. **Select** the best solution — the correspondence set with the greatest consensus — and treat its inliers as the final inlier set.


## What are the big ideas covered by this project?

At the graduate level, Computer Vision is fundamentally concerned with recovering structure, geometry, and correspondence from images — tasks almost always complicated by noise, occlusion, and erroneous measurements. RANSAC sits at the heart of this challenge. RANSAC appears in virtually every classical geometric vision pipeline, from homography estimation to 3-D reconstruction. 

It is not merely a practical tool but a conceptual framework for robust estimation: the idea that a reliable solution can be recovered even when a large fraction of the data is corrupted. Studying RANSAC also builds intuition about probabilistic reasoning, specifically how to trade determinism for confidence guarantees, a mode of thinking that recurs throughout computer vision research. It also provides an essential baseline: many modern estimation methods are benchmarked against RANSAC-based pipelines.

## Long View

I am virtually attending an IEEE conference about Computer Vision, WACV 2026 on March 7-11. Going over papers in workshops has inspired me to study the fundamental algorithms and mathematics that are need to understand computer vision. I have to start from where I am - at zero. With this class paper I will have studied an important algorithm used in homography (making a panoramic view from overlapping 2-D images) and structure from motion/3-D reconstruction (using many 2-D images an object from different view-points to render a sparse or dense 3-D image).

Over the summer, I intend to create a small portfolio of projects focussed on classical computer vision that spans geometric estimation, multi-view reconstruction, feature extraction, and optimization — studied not in isolation, but as a coherent pipeline. Doing this even for a simple stereo or panorama task, will help me consolidate understanding far more effectively than studying each algorithm in isolation. Inspired by my reading of some accessible texts such as Davies [2012] and Szeliski [2022], I intend to implementing the following (the exact algorithms under each component may change as I study more but this is the plan for now). 

For fundamentals, I want to study pin-camera theory and the linear algebra of translating 3-D images to 2-D images. Which moves to Epipolar Geometry and Multi-View - tring to stitch the 2-D images to make a panorama or a 3-D image, such as the fundamental matrix, estimated via the normalized 8-point algorithm. It introduces the geometric relationship between two views and it is a prerequisite to understand the math for any reconstruction tasks. 

For Feature Detection and Description, SIFT (Scale-Invariant Feature Transform) is a likely important starting point. It covers scale-space theory, keypoint detection via the Difference of Gaussians, and gradient-based description, and nearly every feature matching pipeline in classical CV traces its lineage here. 

In the area of Geometric Estimation and Robust Fitting, homography estimation is the ideal first project: it cleanly ties together feature matching, the Direct Linear Transform, and RANSAC into a single end-to-end pipeline and directly motivates why robust fitting is necessary. 

Under optimization,  bundle adjustment — the global refinement step in Structure from Motion — is the most important optimization problem in classical vision. Understanding its formulation and the sparsity structure it exploits maybe an important step in working toward understanding 3-D reconstruction.

## Code Design Thoughts

Rather than working with actual images, this project may work entirely with 2-D point correspondences — pairs of (x, y) coordinates, say 2-D or even 3-D graphs, representing matched points between two hypothetical views. This will abstract the underlying image stitching of pixel. Keeping the data as cartesian points eliminates image I/O and warping complexity, allowing me to focus entirely on the algorithm and its theory. This will also allow me to make graphs and visually represent the efficacy with varying threshold `t` and iterations `k` for the same model (fixed `n`).

As outlined in the timeline - I plan on starting small and grow slowly and I am allowing myself double the time (4 weeks) than the exclusive project time in the course module (2-weeks). I plan on making both parts of the project a Test Driven Development (TDD). Each task will be broken down until it is as simple as possible to write and test. Doing this will also allow me to make the project DRY (Don't Repeat Yourself). The many small tested functions will glued to do the larger task.

I learnt from the mid-term project that developing algorithms from pseudocode is very simple in Python. So, I will first develop a Python project with an clear idea that it is to be translated to C.  will try as much as possible to not depend on packages not available in C. 

Then I will translate it into C test-by-test and function-by-function. 

Like the helper bash functions in assignments or the Python helper function for midterm, I will create methods to build and test from terminal. 

Then using these, I will do the 


## Resources Found
[1] Fischler, M. A. and Bolles, R. C. 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24, 6 (June 1981), 381–395. https://doi.org/10.1145/358669.358692.

[2] Szeliski, R. 2022. Computer Vision: Algorithms and Applications (2nd ed.). Springer. ISBN 978-3-030-34371-2. https://doi.org/10.1007/978-3-030-34372-9.

[3] Davies, E. R. 2012. Computer and Machine Vision: Theory, Algorithms, Practicalities (4th ed.). Academic Press. ISBN 978-0-12-386908-1.

[4] Richard Hartley and Andrew Zisserman. 2004. Multiple View Geometry in Computer Vision. Cambridge University Press.

## Timeline: A 4-Week RANSAC Study & Analysis For Simple Models 
* 2-D and polynomial models, but spatial points rather than pixel.
* Language: C 
* Deliverables: Report + Code
* At each step keep notes that will go into report in week 4

### Week 1 — Theory &  Mathematics

* Days 1–2: Foundations and Motivation
* * Begin with the original Fischler & Bolles (1981) paper, focusing on the motivation: why classical least squares fails in the presence of outliers, and what problem RANSAC was designed to solve. Supplement with Section 8.1.4 of Szeliski and Appendix A6 of Davies to get two complementary perspectives — one oriented toward geometric vision, the other toward practical algorithm design.
* Days 3–6: Algorithm Mechanics and Parameter Analysis

### Week 2 - Algorithm (Python) and C Code Design

* Days 7–9: Algorithm/Pseudo Code Design (using Python) and Fitting to a Test Problem
* * Choose a simple, well-understood geometric fitting problem as testbed — fitting a line to 2-D points is ideal, since `n` = 2 and the residual is easy to compute and show graphically.
* * Generate synthetic 2-D point data in C by adding Gaussian noise to inliers and randomly placed outliers. 
* * Implement RANSAC from scratch in Python (with as little library dependencies as possible but will use what I may have available in C: such as rand()). Keep the code modular — separate functions for each step.
* * * Model fitting (line through two points)
* * * residual computation, and 
* * * consensus counting.
* Days 10-14: Implementation of the Pseudocode/Python Code in C

### Week 3 — Analysis, Experimentation, and Writing

* Days 15–17: Controlled Experiments
* * Set up Python or bash helper files to run experiments.
* * Run a series of experiments for a 2-D model (fixed `n` and `N`) by varying three parameters independently: the outlier fraction or the threshold `t` and the number of iterations `k`. 
* * For each experiment, record whether RANSAC finds the correct model that we created the data with.
* * Analyze how time and memory complexity scale with the dataset size `N` and the iteration count `k`.

* Days 18–21: Complexity Analysis Across Model Types
* * Extend implementation beyond line fitting to at least two progressively more complex models.
* * For each model, 2-D and the other two more complex, analyze how time and memory complexity scale with the minimum sample size `n`, the dataset size `N` and the iteration count `k`.

### Week 4: Report Writing and Final Steps

* Days 22-24: Code Review, Cleanup
* * Refactor the C code for clarity — add comments, check for edge cases, and ensure the threshold `t` and iteration count `k` are configurable via command-line arguments or a header file. Write a short README describing how to compile and run the code.
* Days 25-28: Finalize report that I have been contributing to through the last 3 weeks and submission.
  