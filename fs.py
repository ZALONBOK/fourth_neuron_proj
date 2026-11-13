def check_grade(grade):
    if grade>=100:
        return "perfect"
    if grade>=55:
        return "passed"
    else:
        return "failed"
result = check_grade(1)
print(result)