import re

names = ['ac4deb55-ad246e57-296a8913-199ddddd-d2bd36e5', '36b83bff-e122758c-f15e4e2f-cef828f1-49fe71d3', '33e3bf71-dd7b1051-4c59b9d9-20e03c40-445bc0f2', 'a4d407bd-1d403175-4282009e-fc8fafe9-fa2e6a26', '51ac2649-91c90218-a98a22a8-42e71e3a-90471e6d']

for name in names:
    print("\n ------------------------------- \n")

    report_path = '../png_files_sample/label/{name}.txt'.format(name=name)
    with open (report_path, "r") as r_file:
        file_read = r_file.read()
        print(file_read)
        report = re.split("[\n:]", file_read)
        for i in range(len(report)):
            report[i] = report[i].strip()

        # print("REPORT", report)

        try:
            index = report.index('FINDINGS')
        except ValueError:
            index = report.index('FINDINGS AND IMPRESSION')
        try:
            index2 = report.index('IMPRESSION')
        except ValueError:
            index2 = len(report)
        sentences = ' '.join(report[index+1:index2]).split('. ')
        for sentence in sentences:
            sentence = sentence.lower().replace('.', '').replace(',', '').split()
            print(sentence)

    print("\n ------------------------------- \n")