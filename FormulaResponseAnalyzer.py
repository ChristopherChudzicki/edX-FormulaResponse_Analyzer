import pandas
import numpy
import calc
import random
import json
import os, time, warnings
import lxml.etree as ET

#Eceptions:
from pyparsing import ParseException
from calc import UndefinedVariable

##################################################
#       Patch calc.py to be a bit more foregiving about function domains
##################################################  
calc.DEFAULT_FUNCTIONS['sqrt'] = numpy.lib.scimath.sqrt #numpy.sqrt(-4) = nan, numpy.sqrt(-4+0j)=2j; numpy.lib.scimath.sqrt(-4)=2j
calc.DEFAULT_FUNCTIONS['arccos'] = numpy.lib.scimath.arccos
calc.DEFAULT_FUNCTIONS['arcsin'] = numpy.lib.scimath.arcsin
#my_eval_power is the same as calc.eval_power, except it uses numpy.lib.scimath.power(b,a) instead of b**a. That way (-4)^0.3 will not throw errors.
def my_eval_power(parse_result):
    """
    Take a list of numbers and exponentiate them, right to left.

    e.g. [ 2, 3, 2 ] -> 2^3^2 = 2^(3^2) -> 512
    (not to be interpreted (2^3)^2 = 64)
    """
    # `reduce` will go from left to right; reverse the list.
    parse_result = reversed(
        [k for k in parse_result
         if isinstance(k, calc.numbers.Number)]  # Ignore the '^' marks.
    )
    # Having reversed it, raise `b` to the power of `a`.
    power = reduce(lambda a, b: numpy.lib.scimath.power(b,a), parse_result)
    return power
calc.eval_power = my_eval_power

def ensure_dir(path):
    '''Check if path's directory exists, create if necessary. From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary'''
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        message = "Creating directory {directory}".format(directory=directory)
        warnings.warn(message, RuntimeWarning)
        os.makedirs(directory)

def split_csv_by_problem_id(input_csv_path, problem_id_front, sep='\t', acceptable_response_types=['formularesponse']):
    '''
    (single csv)->many csv
    `input_csv` should contain problem_check table for every problem in the course. This splits the csv into many new csv, one per problem.
    
    input_csv:
    time,   hashed_username,    problem_id,     correctness,    submission,     attempt_number,     input_type,     response_type
    
    output_csv: (one per unique problem_id)
    time,   hashed_username,   submission,     correctness,    attempt_number,     input_type,  response_type
    
    csv are tab-separated by default
    '''
    df = pandas.read_csv(input_csv_path, sep=sep)
    df_filter = df.response_type.isin(acceptable_response_types)
    df = df[ df_filter ]
    grouped = df.groupby("problem_id")
    output_path_template = "problem_check/{problem_name}.csv"
    for group_name, group_df in grouped:
        problem_name = group_name.replace(problem_id_front,"")
        output_csv_path = output_path_template.format(problem_name=problem_name)
        group_df = ProblemCheck(group_df)
        group_df.metadata['problem'] = problem_name
        group_df.export_csv(output_csv_path)
    return
        

def eval_and_summarize(case_sensitive=True, n_evals=5):
    raw_dir = "problem_check"
    eval_dir = "problem_evaluated"
    summary_dir = "problem_summary"
    
    listdir = os.listdir(raw_dir)
    total = len(listdir)
    
    for index, filename in enumerate(listdir):
        raw_path = "{directory}/{filename}".format(directory=raw_dir, filename=filename)
        eval_path = "{directory}/{filename}".format(directory=eval_dir, filename=filename)
        summary_path = "{directory}/{filename}".format(directory=summary_dir, filename=filename)
        
        start = time.time()
        print("\n" + "#"*100 + "\n# Working on {filename} ... {index}/{total}".format(filename=filename, total=total, index=index+1))
        try:
            problem = ProblemCheck.import_csv(raw_path)
            problem.drop_duplicates()
            problem.evaluate(n_evals=n_evals)
            problem.export_csv(eval_path)
            dur = round( (time.time()-start)/60.0 , 2)
            print("#"*10 + " "*10 + "Successfully Evaluated! Runtime: {dur}min".format(dur=dur))
            start = time.time()
            summary = problem.summarize()
            summary.export_csv(summary_path)
            dur = round( (time.time()-start)/60.0 , 2)
            print("#"*10 + " "*10 + "Successfully Summarized! Runtime: {dur}min".format(dur=dur))
            start = time.time()
            summary.make_gui()
            dur = round( (time.time()-start)/60.0 , 2)
            print("#"*10 + " "*10 + "Successfully Constructed GUI! Runtime: {dur}min".format(dur=dur))
        except ParseException:
            dur = round( (time.time()-start)/60.0 , 2)
            print("Could not parse submissions in file. Runetime: {dur}min".format(dur=dur))
        except Exception as e:
            print("!"*50 + "\n" + str(e) + "\n" + "!"*50)
class EmptySubmissionException(Exception):
    """A custom exception we throw when trying to evaluate empty submissions."""
    pass

class ProblemCheckDataFrame(pandas.DataFrame):
    """Parent class, just used so ProblemCheck and ProblemCheckSummary can both inherit these methods
    """
    def __init__(self, *args,**kwargs):
        super(ProblemCheckDataFrame, self).__init__(*args,**kwargs)
        self.from_path = None
        self.metadata = {}
        # specify how empty submissions are encoded. Should be a string
        self.metadata['empty_encoding'] = "EmptySubmission"
        # if this object was instantiated using another ProblemCheckDataFrame, try to get its metadata
        if isinstance(args[0], ProblemCheckDataFrame):
            self.metadata.update(args[0].metadata)
    def _export_metadata(self, filepath):
        """Inserts a two-line header at top of filepath that stores self.metadata
        Header lines are preceded by '#'.
        self.metadata is encoded as JSON
        """
        header = "# {metadata} \n" + "#"*50 + "\n"
        header = header.format(metadata = json.dumps(self.metadata))
        with open(filepath, 'r') as original:
            data = original.read()
        with open(filepath, 'w') as modified:
            modified.write(header + data)
        return
    def _import_metadata(self,filepath):
        """Tries to read metadata from a two-line header at top of filepath.
        Header lines are preceded by '#'.
        metadta is stored as JSON
        """
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if first_line[0]=="#":
                # Remove the '#' and parse as JSON
                metadata = json.loads(first_line[1:])
            else:
                metadata = {}
        # update the current value of self.metadata using values read from CSV. (This keeps defaults if no values are read from CSV). 
        self.metadata.update(metadata)
    def _get_eval_columns(self):
        """Get all columns whose names begin with 'eval'. """
        return [val for val in self.columns.values if val.startswith('eval')]
    @classmethod
    def import_csv(cls, *args, **kwargs):
        """For importing ProblemCheck CSV files.
        
        Same as pandas.read_csv, except:
            0. bound to ProblemCheckDataFrame
            1. tries to read metadata from top of file and binds to object
            2. sep = '\t' by default
            3. stores path of original csv
            4. converts dataframe to object of class cls
            5. retores complex numbers for all columns beginning with 'eval' (see below)
        """
        # add some defaults
        kwargs.setdefault('sep', '\t')
        kwargs.setdefault('comment', '#')
        # Now import the dataframe and convert to class cls
        df = pandas.read_csv(*args,**kwargs)
        df = cls(df)
        # store path
        df.from_path = args[0]
        # Get metadata
        df._import_metadata(df.from_path)
        
        # Fix complex numbers ...
        # Pandas does not import complex numbers properly from csv. This is a reported issue: https://github.com/pydata/pandas/issues/9379
        for eval_col in df._get_eval_columns():
            if df[eval_col].dtype==object: 
                df[eval_col] = df[eval_col].apply(complex)
        
        return df 

    def export_csv(self,*args,**kwargs):
        """For exporting ProblemCehck dataframes.
        
        Same as pandas.DataFrame.to_csv, except:
            0. stores metadata at top of csv
        """
        # Set some defaults
        kwargs.setdefault('sep', '\t')
        # Export the dataframe as csv and add metadata
        try:
            self.to_csv(*args,**kwargs)
        except IOError:
            ensure_dir(args[0])
            self.to_csv(*args,**kwargs)
            
        self._export_metadata(args[0])
        return
class ProblemCheck(ProblemCheckDataFrame):
    """This class stores data and functions associated with analyzing submissions to edX FormulaResponse questions. 
    
    Typical Usage:
        
        Numerically evaluate submission data stored in a CSV file
        >>> example = ProblemCheck.import_csv('example_problem.csv') #load data from CSV 
        >>> example.drop_duplicates()                                #drop duplicate submissions
        >>> example.evaluate()                                       #numerically evaluate submissions
        >>> example.export_csv('example_problem_evaluated.csv')      #export evaluated CSV
        See `evaluate()` for more information
    
        Group (summarize) evaluated submissions by approximate numerical equivalence
        >>> evaluated = ProblemCheck.import_csv('example_problem_evaluated.csv')
        >>> summary = evaluated.summarize()
        >>> summary.export_csv('example_problem_summarized.csv')
        see `summarize()` for more information \
    
    ProblemCheck object can be created in two ways:
        If data is stored as a pandas.DataFrame `data_df`
            >>> ProblemCheck(data_df)
        If data is stored in a csv `data.csv`:
            >>> ProblemCheck.import_csv(`data.csv`)
    
    Either way, data is required to have the following columns:

    hashed_username,   submission,     correctness,  response_type
    str                str             boolean       str
    """
    def __init__(self, *args,**kwargs):
        #Initialize super class
        super(ProblemCheck, self).__init__(*args,**kwargs)
        #Esnure required columns exist
        assert 'hashed_username' in self.columns
        assert 'submission' in self.columns
        assert 'correctness' in self.columns
        assert 'response_type' in self.columns
        self._clean()
        self.metadata['n_submissions'] = self.shape[0]
        self.metadata['n_empty'] = sum(self['submission']==self.metadata['empty_encoding'])
    def _clean(self):
        # If *ALL* submissions to a problem are numeric, pandas will import as numbers not strings and emptys become nan. Replace these by empty_encoding
        self['submission'].fillna(self.metadata['empty_encoding'], inplace=True)
        #After dealing with above very special case, ensure string
        self['submission'] = self['submission'].astype(str)
        #Remove all whitespace
        self['submission'] = self['submission'].replace(r"\s+", '', regex=True)
        #Remove anything of the form `[...]`, for example `[tab]`
        self['submission'] = self['submission'].replace(r'\[.*\]', '', regex=True)
        #Replace empty (or all-whitespace) submissions by empty_encoding
        self['submission'] = self['submission'].replace('', self.metadata['empty_encoding'])
    def drop_ducplicates(self):
        """Drops duplicate submission strings by the same user, records some metadata.
        
        For example, if:
        
            hashed_username    submission
            Bob                'a + b'      # keep
            Bob                'b + a'      # keep
            Bob                'a + b'      # drop
            Bob                'a*b'        # keep
        """
        self.drop_duplicates(subset=["hashed_username","submission"], inplace=True)
        self.metadata['drop_duplicates'] = True
        self.metadata['n_submissions'] = self.shape[0]
        self.metadata['n_empty'] = sum(self['submission']==self.metadata['empty_encoding'])
        return
    def _update_vars_dict_list(self, submission):
        '''
        Detects variables used in submission and updates vars_dict_list accordingly.
        '''
        try:
            # Extract new variables from submission
            submission = calc.ParseAugmenter(submission)
            submission.parse_algebra()
            # Find out which variables are new
            old_vars = set(self.metadata['vars_dict_list'][0].keys())
            full_vars = submission.variables_used
            if not self.metadata['case_sensitive']:
                old_vars = set([var.lower() for var in old_vars])
                full_vars = set([var.lower() for var in full_vars])
            new_vars = full_vars - old_vars
            # Assign values to each new variable
            for var in list(new_vars):
                for vars_dict in self.metadata['vars_dict_list']:
                    vars_dict[var] = random.uniform(0.01,1)
            return 
        except ParseException:
            print("Failed to update varDict")
            return
    
    def _evaluate_empty_row(self, index, row):
        """If a submission is empty, make all of its evaluations numpy.inf"""
        for sample_index, self.vars_dict in enumerate(self.metadata['vars_dict_list']):
            self.loc[index,'eval.'+str(sample_index) ] = numpy.inf 
        return
    
    def _evaluate_row(self, index, row):
        # funcs_dict is used as a required positional argument in the formularesponse grader calc.evaluator. However, it is never used for anything.
        funcs_dict = {}
        if row['submission']==self.metadata['empty_encoding']:
            raise EmptySubmissionException
        else:
            for sample_index, vars_dict in enumerate(self.metadata['vars_dict_list']):
                self.loc[index,'eval.'+str(sample_index) ] = calc.evaluator(vars_dict, funcs_dict, row['submission'],case_sensitive=self.metadata['case_sensitive'])
        return
    
    def evaluate(self, n_evals=2, case_sensitive=True):
        ##################################################
        # Store some metadata
        ##################################################
        self.metadata['n_evals'] = n_evals
        self.metadata['case_sensitive'] = case_sensitive
        # vars_dict_list is alist of dictionaries mapping variable:value for numerically evaluating formulas
        self.metadata['vars_dict_list'] = [{} for j in range(0,self.metadata['n_evals'])]
        ##################################################
        # Do the evaluations
        ##################################################
        for index, row in self.iterrows():
            try:
                self._evaluate_row(index,row)
            except UndefinedVariable as undefined_variable:
                print("Attempting to update vars_dict_list to include {var}".format(var=undefined_variable))
                self._update_vars_dict_list(row['submission'])
                self._evaluate_row(index, row)
            except EmptySubmissionException:
                self._evaluate_empty_row(index, row)
            except ParseException:
                print("Invalid submissions detected: \n" +"#"*25+ "\n" + str(row)) + "\n"+"#"*25
                raise
        return

    @staticmethod
    def _sig_round(z, n_sig=3, chop_past=12):
        z_copy = numpy.copy(z)
        z_copy = numpy.round(z_copy, decimals=chop_past)
        def round_part(x):
            x.flags.writeable=True
            power = 10.0**(-numpy.floor(numpy.log10(abs(x[numpy.flatnonzero(x)]))) + n_sig - 1)
            x[numpy.flatnonzero(x)] = numpy.round(x[numpy.flatnonzero(x)]*power,0)/power
            return x
        return round_part(z_copy.real) + round_part(z_copy.imag)*1j        
    
    def summarize(self, n_sig=3):
        """Summarize an evaluated problem check table.
        
        INPUT data must be evaluated first. I.e., these columns are required:
            
            submission,  correctness,  eval.0,  ...,  eval.(n_eval-1), response_type
            string       0/1           float         float              string
    
        OUTPUT data has statistics on these columns, plus response_type.
        
            Output statistics come in two types:
                
                1. eval: aggregated over submissions with numerically equivalent evaluations (to n_sig sig figs)
                2. subm: aggregated over submissions with identical submission strings
            
            subm_correctness: of identical submissions, what fraction were marked correct? [should be 0 or 1, might not be because of edX grader random samples]
            subm_count: number of submissions with identical submission strings 
            eval_correctness: of submissions with numerically equivalent evals, what fraction were marked correct? [should be 0 or 1, might not be because of edX grader random samples]
            eval_count: number of submissions with equivalent evaluations
            subm_frequency: this string's frequency among numerically equivalent submissions
            eval_frequency: fraction of total submissions with this numerical evaluation
        """
        eval_cols = self._get_eval_columns()
        # round eval columns to specified number of sig figs, then convert to floats so pandas.DataFrame.groupby can sort them
        rounded_df = self.copy()
        for eval_col in eval_cols:
            rounded_df[eval_col] = self._sig_round(rounded_df[eval_col],n_sig=n_sig)
            rounded_df[eval_col] = rounded_df[eval_col].apply(str)
        
        #########################
        # Group and Summarize
        #########################
        # Two submissions will go in the same group if all rounded evaluations are same and submission strings are same
        subm_grouped = rounded_df.groupby(by=eval_cols+['submission',])    
        summary = subm_grouped.aggregate({
                     'correctness':numpy.mean,
                    }, sort=False)
        
        # correctness is averaged over submission groups. Rename and convert to float.
        summary.rename(columns={'correctness': 'subm_correctness'}, inplace=True)
        summary['subm_correctness'] = summary['subm_correctness']*1.0    
        # Size of submission groups
        summary['subm_count'] = subm_grouped.size()
        #Size of everyone
        total = sum(summary['subm_count'])
        # Size of eval groups
        def eval_group_size(eval_group):
            return sum(summary.loc[eval_group,'subm_count'])
        
        eval_groups = summary.groupby(level=eval_cols).groups.keys()
        # now sort the eval groups by size:
        eval_groups.sort(key = eval_group_size, reverse=True)
        
        for index, eval_group in enumerate(eval_groups):
            summary.loc[eval_group,'eval_group'] = index
            summary.loc[eval_group,'eval_count']= eval_group_size(eval_group)
            # percent of submissions that **I have declared** numerically equivalent that were graded correct by edX
            summary.loc[eval_group,'eval_correctness']=numpy.mean(summary.loc[eval_group,'subm_correctness'])
        # eval group frequencies: percent of ALL submissions that evaluated numerically equiv
        summary['eval_frequency'] = summary['eval_count']/total
        # submission group frequencies: percent of numerically equivalent submssions that are same string
        summary['subm_frequency']  = summary['subm_count']/summary['eval_count']
        summary['response_type'] = rounded_df['response_type'].iloc[0]
        summary.sort_values(by=['eval_count','subm_count'],inplace=True,ascending=False)
        
        # At this point, {eval.0,eval.1, ... submission} are row indices not columns. Let's restore them to columns
        summary.reset_index(inplace=True)
        # Convert summary to a ProblemCheckSummary object
        summary = ProblemCheckSummary(summary)
        # bind metadata
        summary.metadata.update(self.metadata)
        
        return summary

class ProblemCheckSummary(ProblemCheckDataFrame):
    """docstring for ProblemCheckSummary"""
    def __init__(self, *args, **kwargs):
        #Initialize super class
        super(ProblemCheckSummary, self).__init__(*args,**kwargs)
    def make_gui(self):
        """Make a a single-problem GUI from problem summary.
        
        GUI is an HTML file written to /gui/problem/problem_name.csv.
        
        Dependent Files:
            gui/resources/gui_problem.css
            gui/resources/gui_problem.js
        
        GUI Structure is:
        
            `2*a*b` ... ... group statistics            <-- summary
                `2*a*b` ... submission statistics       <-- details
                `2*b*a` ... submission statistics       <-- details
                `a*b*2` ... submission statistics       <-- details
                `a*2*b` ... submission statistics       <-- details
            `a^2`   ... ... group statistics            <-- summary
                `a^2`   ... submission statistics       <-- details
                `a*a`   ... submission statistics       <-- details
                `a^3/a` ... submission statistics       <-- details
            etc
        """
        data_columns = [
            {
                'gui':'Submission',     # GUI column title
                'summary':'submission', # summary row data source
                'details':'submission'  # details row data source
            },
            {
                'gui':'Correctness',
                'summary':'eval_correctness',
                'details':'subm_correctness'
            },
            {
                'gui':'Frequency',
                'summary':'eval_frequency',
                'details':'subm_frequency'
            },
            {
                'gui':'Count',
                'summary':'eval_count',
                'details':'subm_count'
            }
        ]
        ##################################################
        # Helper Functions
        ##################################################
        def get_gui_template():
            template_path = "gui/templates/gui_problem_template.html"
            with open(template_path,'r') as f:
                parser = ET.XMLParser(remove_blank_text=True)
                tree = ET.parse(f,parser)
            return tree.getroot()
        def get_group_summary_data(group_df, data_columns):
            summary_cols = []
            details_cols = []
            gui_cols = []
            for d in data_columns:
               summary_cols.append( d['summary'] )
               gui_cols.append( d['gui'] )
        
            # Use first row of df to get summary values. This will return a Series not a DataFrame, so convert it to a DataFrame and transpose.
            summary = group_df[summary_cols].iloc[0]
            summary = pandas.DataFrame(summary).transpose()
            # Rename columns
            summary.columns = gui_cols
        
            return summary
    
        def get_group_details_data(group_df, data_columns):
            details_cols = []
            gui_cols = []
            for d in data_columns:
               details_cols.append( d['details'] )
               gui_cols.append( d['gui'] )
        
            details = group_df[details_cols]
            details.columns = gui_cols
        
            return details
        
        
        def insert_group_html(gui_html, group_df, data_columns):
            # Get summary and details data frames
            summary = get_group_summary_data(group_df, data_columns)
            details = get_group_details_data(group_df, data_columns)
            # Get summary and details HTML strings. These have structure:
            # <table>
            #   <tbody>
            #       DATA
            #   </tbody>
            # </table>
            # We just want the tbody, so we get the [0]th child of table. 
            summary_tbody = ET.fromstring(summary.to_html(header=False,index=False))[0]
            details_tbody = ET.fromstring(details.to_html(header=False,index=False))[0]
            # Add classes
            summary_tbody.attrib['class'] = "summary"
            details_tbody.attrib['class'] = "details"
            # Append summary and details to main table.
            table = gui_html.find('body/table')
            table.extend([summary_tbody, details_tbody])
            return
        
        def export_gui_html(gui_html):
            output_html_path = "gui/problem/{problem}.html".format(problem=self.metadata['problem'])
            try:
                with open(output_html_path,'w') as f:
                    ET.ElementTree(gui_html).write(f, pretty_print=True, method="html")
            except IOError:
                ensure_dir(output_html_path)
                with open(output_html_path,'w') as f:
                    ET.ElementTree(gui_html).write(f, pretty_print=True, method="html")
            return
        ##################################################
        # Make the GUI
        ##################################################
        gui_html = get_gui_template()
        grouped = self.groupby(by='eval_group')
        for group, df in grouped:
            insert_group_html(gui_html, df, data_columns)
        
        export_gui_html(gui_html)
        return


# test_path = "problem_summary/HW_1.3.csv"
# df = ProblemCheckSummary.import_csv(test_path)
# df.make_gui()
# print(df)
# x = df.iloc[0,:]
# print(x.to_frame().transpose().to_html())