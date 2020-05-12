import re

class CXRReports:
    def __init__(self, path):
        self.data_path = path
        self.reports = []
        self.idx = 0

        for line in open(self.data_path+'_reports.txt'):
            self.reports.append(line.strip())

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.reports)

    def __next__(self):
        if self.idx >= len(self.reports):
            raise StopIteration
        report = self.reports[self.idx]
        findings = get_report_findings(report)
        words = []
        for finding in findings:
            tokens = finding.lower().replace('.', '').replace(',', '').split()
            for word in tokens:
                words.append(word)
        self.idx += 1
        return words


def get_report_findings(report_path):
    try:
        with open (report_path, "r") as r_file:
            file_read = r_file.read()
            report = re.split("[\n:]", file_read)
            for i in range(len(report)):
                report[i] = report[i].strip().lower()

        try:
            index = report.index('findings')
        except ValueError:
            index = report.index('findings and impression')
        try:
            index2 = report.index('impression')
        except ValueError:
            index2 = len(report)

        sentences = ' '.join(report[index+2:index2]).split('. ')
    except ValueError:
        sentences = []

    return sentences
