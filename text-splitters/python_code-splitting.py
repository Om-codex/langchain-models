from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade, courses=None):
        self.name = name
        self.age = age
        self.grade = grade
        self.courses = courses if courses is not None else []

    def enroll(self, course):
        if course not in self.courses:
            self.courses.append(course)
            return f"{self.name} enrolled in {course}."
        return f"{self.name} is already enrolled in {course}."

    def info(self):
        return (f"Name: {self.name} | Age: {self.age} | Grade: {self.grade} "
                f"| Courses: {', '.join(self.courses) if self.courses else 'None'}")


# Example object
student1 = Student("Alice Johnson", 20, "A", ["Math", "Physics"])
print(student1.info())
student1.enroll("Chemistry")
print(student1.info())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
  language = Language.PYTHON,
  chunk_size = 750,
  chunk_overlap  = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])