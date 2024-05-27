import timeit
import re

# Existing regex pattern
arabic_letters_pattern = re.compile(
    r'[\u0621-\u063A\u0641-\u064A\u0660-\u0669\u0671-\u06D3\u06FA-\u06FF]'
)

def count_arabic_letters(word):
    return len(re.findall(arabic_letters_pattern, word))

def manual_count_arabic_letters(word, arabic_alphabet):
    return sum(1 for char in word if char in arabic_alphabet)

# Example word for testing
word = "السّلام عليكم"
arabic_alphabet_set = set([
    "\u0621", "\u0622", "\u0623", "\u0624", "\u0625", "\u0626", "\u0627", "\u0628",
    "\u0629", "\u062A", "\u062B", "\u062C", "\u062D", "\u062E", "\u062F", "\u0630",
    "\u0631", "\u0632", "\u0633", "\u0634", "\u0635", "\u0636", "\u0637", "\u0638",
    "\u0639", "\u063A", "\u0641", "\u0642", "\u0643", "\u0644", "\u0645", "\u0646",
    "\u0647", "\u0648", "\u064A", "\u0660", "\u0661", "\u0662", "\u0663", "\u0664",
    "\u0665", "\u0666", "\u0667", "\u0668", "\u0669", "\u0671", "\u067E", "\u0686",
    "\u06A4", "\u06AF", "\u06CC", "\u06D5", "\u06FA", "\u06FF"
])

# Benchmarking
regex_time = timeit.timeit(lambda: count_arabic_letters(word), number=100000)
manual_time = timeit.timeit(lambda: manual_count_arabic_letters(word, arabic_alphabet_set), number=100000)

print(f"Regex method time: {regex_time:.6f} seconds")
print(f"Manual method time: {manual_time:.6f} seconds")
