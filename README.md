# FormulaResponse_Analyzer.py
Python script for analyzing submissions to edX formularesponse problems. Goals:

1. For each formularesponse problem in the course, group submissions by mathematical equivalence.
2. Provide an HTML/JS based user interface for viewing the grouped submissions.
3. For an entire course, provide an HTML/JS interface to help course team identify problems that:
    * would benefit from wrong-answer feedback
    * might have grading issues (e.g., poorly set numerical sampling range)

`Formularesponse_Analyzer.py` builds a problem index for the entire course and summary page for each problem. Summary page for a single problem looks like this:
![output of FormulaResponse_Analyzer for one problem](https://github.com/ChristopherChudzicki/edX-FormulaResponse_Analyzer/blob/master/problem_output_screenshot.png "Problem Output")

##Usage
`Formularesponse_Analyzer.py` requires as input a `csv` file that contains submission data for each problem and user in your course. The file must contain columns for `hashed_username`, `correctness`, `submission`, and `response_type`. The "problem_checks_augmented" table from MITx is sufficient:

|                     |                 |                                    |             |            |                |                      |                 | 
|---------------------|-----------------|------------------------------------|-------------|------------|----------------|----------------------|-----------------| 
| time                | hashed_username | problem_id                         | correctness | submission | attempt_number | input_type           | response_type   | 
| 2015-05-13T08:28:21 | user1           | i4x-MITx-FakeCourse-problem-HW_1.1 | TRUE        | sqrt(x-y)  | 3              | formulaequationinput | formularesponse | 
| 2015-05-13T08:28:22 | user1           | i4x-MITx-FakeCourse-problem-HW_1.1 | FALSE       | x-y        | 2              | formulaequationinput | formularesponse | 
| 2015-05-13T08:28:23 | user1           | i4x-MITx-FakeCourse-problem-HW_1.1 | FALSE       | sqrt(y-x)  | 1              | formulaequationinput | formularesponse | 
| 2015-05-13T08:28:24 | user2           | i4x-MITx-FakeCourse-problem-HW_1.1 | TRUE        | sqrt(x-y)  | 2              | formulaequationinput | formularesponse | 
| ...                 | ...             | ...                                | FALSE       | ...        | ...            | ...                  | ...             | 

(A sample datafile is included in this repository, `MITx_FakeCourse_problem_checks_augmented.csv`)

Then: 

1. Place the `problem_checks_augmented.csv` in the Formularesponse_Analyzer folder.
2. Edit "Run Stuf" portion of `FormulaResponse_Analyzer.py` as necessary. You'll probably need to specify:
    * `problem_checks_augmented_filename`
    * `problem_id_front` 

## Background
It is possible within within edX to give wrong-answer feedback on formularesponse ("symbolic input") problems using the hintgroup tag. However, existing edX analytics tools (Insights and the MITx dashboard) group formularesponse submissions by string equivalence rather than mathematical equivalence (e.g., x+x and 2x are grouped separately) which makes identifying common answers somewhat difficult. 

Additionally, formularesponse problems sometimes have grading issues: the edX formularesponse grader checks for equivalence to the instructor provided answer by numerical sampling and instructors must set the sample range manually for each problem. This can lead to several issues:

* Samples may be outside the domain for mathematical functions used in correct submissions. This can cause all correct answers to me marked incorrect, or can cause identical strings to be graded differently for different users. Because edX uses different random samples for each user, if the sampe range for x is x\in[0,2] then the string sqrt(1-x) may be marked as correct for some users but incorrect for others.
* wrong answers may be erroneously marked correct (e.g., x and sin(x) will be graded the same if the x samples are small)
* correct answers may be marked wrong (e.g., in a physics problem cos(x) and sqrt(1-sin(x)) may be the same in a physics problem because x is in the first quadrant, but if grading samples are taken from all quadrants, these answers will be treated differently )

## Functions in FormulaResponse_Analyzer.py
FormulaReponse_Analyzer.py has five steps, each implemented as a python function. Intermediate information is written to a new `csv` at the end of each step. Below are some overall comments on these functions. For details about parameters, inputs, and outputs, see the extensive code comments.

1. `split_csv_by_problem_id`:
 * splits single original tab-delimited `csv` file with all users and all problems into into one file per problem
 * Some problems are skipped based on input types. We're mainly interested in symbolic problems, so we skip all problems except those whose input type is `textline`and `formulaequationinput`. The resulting collection of problems includes many non-symbolic problems.(numerical and customresponse problems often use `textline`).
2. `make_eval_csv`:
 * numerically evaluates the submissions to a single problem a specified number of times. Evaluation is performed using edX calc.py library
 * no information is read from problem XML. We detect variables and assign random samples within the range [0,1].
 * `calc.py` is patched to use `numpy.lib.scimath.sqrt` instead of `numpy.sqrt` to avoid sampling errors. [`numpy.sqrt(-4) = NaN`; `numpy.lib.scimath.sqrt(-4)=2j`]. A few other similar functions (e.g., `arcsin` and `power`) are also patched.
 * Evaluations can be complex (e.g., `arcsin(5)` ).
 * By default, evaluations are case-insensitive. (This is the edX default).
 * exact duplicate submissions by the same user are skipped by default.
 * data is handled using the pandas python module 
3. `make_summary_csv`:
 * group problems by equivalent evaluations. “Equivalent” means numerically identical on all evaluations after rounding to a specified number of significant digits.
 * This grouping procedure is obvious not perfect, but seems to work very well within 8MechCx and is very easy to implement using the built-in groupby method in pandas.  
 * for each group, calculate various statistics including 
 * frequency of equivalent (by evaluation) submissions
 * frequency of identical (by string) submissions
 * fraction of submissions we believe are equivalent that edX graded as correct [should be 0 or 1]
 * fraction of identical submissions that edX graded as correct [should be 0 or 1]
4. `make_gui_problem_table`:
 * summarize submissions for a single problem. This is an HTML file built with  the ElementTree python module [plus some javascript]
5. `make_gui_index`:
 * This is an HTML file built with  the `lxml.etree` python module 
 * index of problem summaries for every formularesponse problem in the course
 * In order to make finding “interesting” problems easy, every formularesponse problem has three statistics:
    * Feedback Score (range, 0–1): The wrong answers form a probability distribution. This is just the second moment of that distribution, errorsperror2. This second moment is larger if the wrong answers are of just a few common types. This sort of problem would benefit more from wrong-answer feedback
    * n_correct : the number of submission groups wherein some but not all submissions were marked as correct. Should be 1. If bigger than 1, could indicate grader is not strict enough.
    * n_partial: number of submission groups wherein some but not all submissions were marked as correct. Should be 0. If not zero, could indicate a variety of grading issues.
Index is sortable (using `tablesorter.js`)


## Problem XML
Currently FormulaResponse_Analyzer does not get any information from the problem XML for two reasons:

1. we believe problem XML would be difficult (certainly not impossible) to parse and match to tracking logs. [e.g., XML names and tracking logs names don’t always match] 
2. although problem XML is available to course authors, we do not believe it is available to analytics dashboards

Not using information from the problem XML presents several changes including:

 * we don’t know what variables are expected (this is fairly easy to solve)
 * we do not know if the problem was graded by edX as case sensitive or case insenstive (edX default is case insensitive)

This is not a big problem, as long as the course authors are consistent
we do not know the numerical sampling range intended by the author. [Using a different sample range than the edX grader can be advantageous, as our analysis can be used to validate edX grading and detect problems with inappropriate author set sample ranges]
