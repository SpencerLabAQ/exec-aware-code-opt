import xml.etree.ElementTree as ET
from text_utils import *

import re
import logging

def get_trace(log_file, lang):
    tree = ET.parse(log_file)
    timed_out = False
    trace = tree.getroot()
    current_lineno = None
    states = []
    current_state = []
    lineno = None
    variable_list_states = {}
    any_modified = False
    # TODO: handle calls within line
    for child in trace:
        assert child.tag == "line", f"'{child.tag}' should be 'line'"
        lineno = int(child.attrib["line"])
        for variable in child:
            if variable.tag == "variable":
                age = variable.attrib["age"]
                name = variable.attrib["name"]
                type = variable.attrib["typ"]

                if age in ("new", "modified"):
                    if age == "modified":
                        any_modified = True
                    val = get_real_text(variable.text, lang)
                    val_text = val
                    current_state.append((age, type, name, val_text))
                    variable_list_states[name] = val
            elif variable.tag == "timeout":
                timed_out = True
        if lineno != current_lineno:
            states.append((lineno, current_state))
            current_state = []
    return states, any_modified, timed_out

def classify_type(c_type):
    # Regular expressions for matching patterns
    array_pattern = r"\[\d*\]"  # Matches arrays like [3], [5], etc.
    pointer_pattern = r"\*"  # Matches pointers like *

    # Check for array pattern
    if re.search(array_pattern, c_type):
        return "array"
    
    # Check for pointer pattern
    if re.search(pointer_pattern, c_type):
        return "pointer"

    if "::" in c_type:
        return "class"
    
    # If not array or pointer, consider it as basic type
    return "basic_type"

def quantize_value(data_type, value_type, value, age):
    try:
        # basic_type int
        if data_type == "basic_type" and value_type.lower() in ("long", "int"):
            value = int(value)
            if 0 < value < 10000:
                return "POSITIVE-REG"
            elif value > 10000:
                return "POSITIVE-VL"
            elif value == 0:
                return "ZERO"
            elif -10000 < value < 0:
                return "NEGATIVE-REG"
            elif value < -10000:
                return "NEGATIVE-VL"
        
        # basic_type Float/Double
        elif data_type == "basic_type" and value_type.lower() in ["float", "double"]:
            value = float(value)
            if 0.0 < value <= 1.0:
                return "POSITIVE-VS"
            elif 1.0 < value < 10000.0:
                return "POSITIVE-REG"
            elif value >= 10000.0:
                return "POSITIVE-VL"
            elif value == 0.0:
                return "ZERO"
            elif -1.0 < value < 0:
                return "NEGATIVE-VS"
            elif -10000.0 <= value < -1.0:
                return "NEGATIVE-REG"
            elif value < -10000.0:
                return "NEGATIVE-VL"
        
        # basic_type Character
        elif data_type == "basic_type" and value_type.lower() == "char":
            if value == '\0':
                return "NULL"
            elif value.isalpha():
                return "ALPHA"
            else:
                return "NOT-ALPHA"
        
        # basic_type Boolean
        elif data_type == "basic_type" and "bool" in value_type.lower():
            value = bool(value)
            if value == 0:
                return "FALSE"
            elif value == 1:
                return "TRUE"
        
        # basic_type Void
        elif data_type == "basic_type" and value_type.lower() == "void":
            return "VOID"
        
        # Array int
        elif data_type == "array" and value_type.lower() == "int":
            if age == "modified":
                # Temporary patch
                return "INIT"
            else:
                return "NOT-INIT"
        
        # array Float/Double
        elif data_type == "array" and value_type.lower() in ["float", "double"]:
            if age == "modified":
                # Temporary patch
                return "INIT"
            else:
                return "NOT-INIT"
        
        # array Character
        elif data_type == "array" and value_type.lower() == "char":
            if age == "modified":
                # Temporary patch
                return "INIT"
            else:
                return "NOT-INIT"
        
        # Pointer
        elif data_type == "pointer":
            if value == "0x0":
                return "NULL"
            else:
                return "NOT-NULL"
        
        # Handle invalid inputs
        return "OTHER"
    except Exception as e:
        logging.error(f"An error occurred during quantization {data_type} {value_type} {value}: {str(e)}")
        if value == "<optimized out>":
            return "OPT"
        return "OTHER"
