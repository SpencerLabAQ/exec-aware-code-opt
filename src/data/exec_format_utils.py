import re

def label_line_cov(count):
    if count > 0:
        return "<e>"
    else:
        return ""

def quantize_exec_count(count):
    assert count >= 0, "The count of line executions has to be a positive number"
    
    if count == 1: 
        return "<e>"
    elif count > 1 and count <= 5:
        return "<e+>"
    elif count > 5 and count <= 20.5:
        return "<E>"
    elif count > 20.5:
        return "<E+>"
    # zero case 
    return ""

def label_branch_exec(branch_exec_val):
    '''
    Three special tokens have been used to label branch coverage:
    - <NB>: Not a branch
    - <BC>: Branch covered in trace
    - <BNC>: Branch not covered in trace
    '''

    if branch_exec_val == None:
        return ""
    elif branch_exec_val == True:
        return "<BC>"
    elif branch_exec_val == False:
        return "<BNC>"
    else:
        raise ValueError(f"'{branch_exec_val}' value not allowed for branch coverage")

def remove_comments(text):
    # The pattern /\*.*?\*/ matches /* followed by any characters including newlines, ending with */
    pattern = r'/\*.+?\*/'
    # re.sub removes the pattern from the text, re.DOTALL allows . to match newlines
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text