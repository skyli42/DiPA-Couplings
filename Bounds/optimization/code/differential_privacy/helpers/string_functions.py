from helpers.constants import Outputs

def superscript(n):
    return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c) - ord('0')] for c in str(n)])


def cleanse_output_history(seq: list[str]):
    """
    Remove all empty strings from seq.
    :param seq: The sequence to cleanse.
    :return: The cleansed sequence.
    """

    return [s for s in seq if s]


def convert_outputs_to_str(seq: list[Outputs]) -> list[str]:
    """
    Converts a list of outputs to a list of strings.
    :param seq: The list of outputs.
    :return: The list of strings.
    """
    return [output.value for output in seq]


def format_list_of_strings(seq: list[str]) -> str:
    out = ""

    cur_symbol = seq[0]
    cur_count = 1

    for i in range(1, len(seq)):
        if seq[i] != cur_symbol:
            out += cur_symbol
            if cur_count > 1:
                out += superscript(cur_count)
            out += ' '
            cur_symbol = seq[i]
            cur_count = 1
        else:
            cur_count += 1

    out += cur_symbol
    if cur_count > 1:
        out += superscript(cur_count)

    return out

def format_input_history(seq: list) -> str:

    if len(seq) == 0:
        return ""

    seq = [str(x) for x in seq]

    return format_list_of_strings(seq)

def format_output_history(seq: list[Outputs]) -> str:

    if len(seq) == 0:
        return ""

    seq = convert_outputs_to_str(seq)

    return format_list_of_strings(seq)
