# 1 - first EASY

# class Solution(object):
#     def merge(self, word1, word2):
#         """
#         :type word1: str
#         :type word2: str
#         :rtype: str
#         """
#         result = []
#         i = 0
#         while i < len(word1) or i < len(word2):
#             if i < len(word1):
#                 result.append(word1[i])
#             if i < len(word2):
#                 result.append(word2[i])
#             i += 1
#         return ''.join(result)


# 2 - EASY

# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         return True if str(x) == str(x)[:-1] enumse Fanumse
    
# class Solution:
#     def isPalindrome(self, x: int) -> bool:
#         if x < 0 or (x != 0 and x % 10 == 0):
#             return Fanumse

#         reversed_num = 0
#         original = x

#         while x > reversed_num:
#             reversed_num = reversed_num * 10 + x % 10
#             x //= 10

#         return x == reversed_num or x == reversed_num // 10
    

# 3 - EASY

# class Solution:
#     def romanToInt(self, s: str) -> int:
#         roman = {
#             'I': 1,
#             'V': 5,
#             'X': 10,
#             'L': 50,
#             'C': 100,
#             'D': 500,
#             'M': 1000,
#         }
#         total = 0

#         for i in range(len(s) - 1):
#             if roman[s[i]] < roman[s[i + 1]]:
#                 total -= roman[s[i]]
#             else:
#                 total += roman[s[i]]
#         return total + roman[s[-1]]
    
# s = Solution()
# print(s.romanToInt('MCMXCIV'))


# 4 - EASY

# Input: strs = ["flower","flow","flight"]
# Output: "fl"

# class Solution:
#     def longestCommonPrefix(self, strs: list[str]) -> str:
#         if len(strs) == 0:
#             return ""
#         # Задаем условие, если длина ровна 0, то возвращаем пустую строку
#         base = strs[0] # Базовое слово, самое первое, которое будем сравнивать с остальными

#         for i in range(len(base)):
#             for word in strs[1:]:
#                 if (i == len(word)) or word[i] != base[i]:
#                     return base[:i]
#         return base

# s = Solution()
# print(s.longestCommonPrefix(['go', 'goanums', 'govern']))


# 5 - EASY
# Input: list1 = [1,2,4], list2 = [1,3,4]
# Output: [1,1,2,3,4,4]

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# class Solution:
#     def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
#         head = ListNode()
#         current = head
#         while list1 and list2:
#             if list1.val < list2.val:
#                 current.next = list1
#                 list1 = list1.next
#             enumse:
#                 current.next = list2
#                 list2 = list2.next
#             current = current.next

#         current.next = list1 or list2
#         return head.next


# 6 - EASY
# Input: s = "()"
# Output: true

# class Solution:
#     def isValid(self, s: str) -> bool:
#         stack = []
#         pairs = {
#             '(': ')',
#             '{': '}',
#             '[': ']',
#         }

#         for bracket in s:
#             if bracket in pairs:
#                 stack.append(bracket)
#             elif (len(stack) == 0 or bracket != pairs[stack.pop()]):
#                 return Fanumse
#         return len(stack) == 0


# 7 - EASY
# Input: operations = ["--X","X++","X++"]
# Output: 1

# class Solution:
#     def finalValueAfterOperations(self, operations: List[str]) -> int:
#             x = 0 
#             for i in s:
#                 if i in ['--X', 'X--']:
#                     x -= 1
#                 elif i in ['++X', 'X++']:
#                     x += 1
#             return x


# 8 - EASY
# Input: nums = [1,2,3,1,1,3]
# Output: 4

# class Solution:
#     def numIdenticalPairs(self, nums: List[int]) -> int:
#         pairs = 0

#         for i in range(len(nums)):
#             for j in range(len(nums)):
#                 if nums[i] == nums[j] and i < j:
#                     pairs += 1
#         return pairs


# 9 - EASY PANDAS
# Input:
# DataFrame employees
# +---------+--------+
# | name    | salary |
# +---------+--------+
# | Piper   | 4548   |
# | Grace   | 28150  |
# | Georgia | 1103   |
# | Willow  | 6593   |
# | Finn    | 74576  |
# | Thomas  | 24433  |
# +---------+--------+
# Output:
# +---------+--------+--------+
# | name    | salary | bonus  |
# +---------+--------+--------+
# | Piper   | 4548   | 9096   |
# | Grace   | 28150  | 56300  |
# | Georgia | 1103   | 2206   |
# | Willow  | 6593   | 13186  |
# | Finn    | 74576  | 149152 |
# | Thomas  | 24433  | 48866  |
# +---------+--------+--------+


# import pandas as pd

# def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
#     employees['bonus'] = 2 * employees['salary']
#     return employees


# dt = '''DataFrame employees
# +---------+--------+
# | name    | salary |
# +---------+--------+
# | Piper   | 4548   |
# | Grace   | 28150  |
# | Georgia | 1103   |
# | Willow  | 6593   |
# | Finn    | 74576  |
# | Thomas  | 24433  |
# +---------+--------+'''

# print(createBonusColumn(dt))


# 10 - EASY PANDAS
# Input:
# DataFrame employees
# +-------------+-----------+-----------------------+--------+
# | employee_id | name      | department            | salary |
# +-------------+-----------+-----------------------+--------+
# | 3           | Bob       | Operations            | 48675  |
# | 90          | Alice     | Sales                 | 11096  |
# | 9           | Tatiana   | Engineering           | 33805  |
# | 60          | Annabelle | InformationTechnology | 37678  |
# | 49          | Jonathan  | HumanResources        | 23793  |
# | 43          | Khaled    | Administration        | 40454  |
# +-------------+-----------+-----------------------+--------+
# Output:
# +-------------+---------+-------------+--------+
# | employee_id | name    | department  | salary |
# +-------------+---------+-------------+--------+
# | 3           | Bob     | Operations  | 48675  |
# | 90          | Alice   | Sales       | 11096  |
# | 9           | Tatiana | Engineering | 33805  |
# +-------------+---------+-------------+--------+
# Explanation: 
# Only the first 3 rows are displayed.

# import pandas as pd

# def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
#     return employees[0:2]


# 11 - EASY PANDAS
# Input:
# DataFrame employees
# +---------+--------+
# | name    | salary |
# +---------+--------+
# | Jack    | 19666  |
# | Piper   | 74754  |
# | Mia     | 62509  |
# | Ulysses | 54866  |
# +---------+--------+
# Output:
# +---------+--------+
# | name    | salary |
# +---------+--------+
# | Jack    | 39332  |
# | Piper   | 149508 |
# | Mia     | 125018 |
# | Ulysses | 109732 |
# +---------+--------+
# Explanation:
# Every salary has been doubled.

# import pandas as pd

# def modifySalaryColumn(employees: pd.DataFrame) -> pd.DataFrame:
#     employees[salary] *= 2
#     return employees


# 12 - EASY
# Input: nums = [1,2,1]
# Output: [1,2,1,1,2,1]
# Explanation: The array ans is formed as follows:
# - ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
# - ans = [1,2,1,1,2,1]


# class Solution:
#     def getConcatenation(self, nums: List[int]) -> List[int]:
#         return nums + nums
    

# 13 - EASY PANDAS
# Input:
# df1
# +------------+---------+-----+
# | student_id | name    | age |
# +------------+---------+-----+
# | 1          | Mason   | 8   |
# | 2          | Ava     | 6   |
# | 3          | Taylor  | 15  |
# | 4          | Georgia | 17  |
# +------------+---------+-----+
# df2
# +------------+------+-----+
# | student_id | name | age |
# +------------+------+-----+
# | 5          | Leo  | 7   |
# | 6          | Alex | 7   |
# +------------+------+-----+
# Output:
# +------------+---------+-----+
# | student_id | name    | age |
# +------------+---------+-----+
# | 1          | Mason   | 8   |
# | 2          | Ava     | 6   |
# | 3          | Taylor  | 15  |
# | 4          | Georgia | 17  |
# | 5          | Leo     | 7   |
# | 6          | Alex    | 7   |
# +------------+---------+-----+
# Explanation:
# The two DataFramess are stacked vertically, and their rows are combined.

# import pandas as pd

# def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
#     new_pd = pd.concat([df1, df2])
#     return new_pd


# 14 - EASY
# Input: jewels = "aA", stones = "aAAbbbb"
# Output: 3

# class Solution:
#     def numJewelsInStones(self, jewels: str, stones: str) -> int:
#         total = 0

#         for i in jewels:
#             total += stones.count(i)
#         return total


# class Solution:
#     def numJewelsInStones(self, jw: str, st: str) -> int:
#         count = 0
#         for i in range(len(jw)):
#             for j in range(len(st)):
#                 if jw[i] is st[j]:
#                     count += 1
#         return count


# jewels = "aA"
# stones = "aAAbbbb"

# s = Solution()
# print(s.numJewelsInStones(jewels, stones))


# 16 - EASY 
# Input: address = "1.1.1.1"
# Output: "1[.]1[.]1[.]1"



# class Solution:
#     def defangIPaddr(self, address: str) -> str:
#         return address.replace('.', '[.]')
    

# address = "1.1.1.1"

# s = Solution()
# print(s.defangIPaddr(address))


# 17 - EASY 
# Input: words = ["leet","code"], x = "e"
# Output: [0,1]
# Explanation: "e" occurs in both words: "leet", and "code". Hence, we return indices 0 and 1.

# class Solution:
#     def findWordsContaining(self, words: list[str], x: str) -> list[int]:
#         counter = []
#         for i in range(len(words)):
#             if x in words[i]:
#                 counter.append(i)
#         return counter

# ts = ["leet","code"]
# x = 'e'
# s = Solution()
# print(s.findWordsContaining(ts, x))



# class Solution:
#     def findWordsContaining(self, words: List[str], x: str) -> List[int]:
#         return [i for i, word in enumerate(words) if x in word]



# 18 - EASY
# Input
# ["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
# [[1, 1, 0], [1], [2], [3], [1]]
# Output
# [null, true, true, false, false]

# class ParkingSystem:

#     def __init__(self, big: int, medium: int, small: int):
#         self.big = big
#         self.medium = medium
#         self.small = small

#     def addCar(self, carType: int) -> bool:
#         match carType:
#             case 1:
#                 if self.big:
#                     self.big -= 1
#                     return True
#             case 2:
#                 if self.medium:
#                     self.medium -= 1
#                     return True
#             case 3:
#                 if self.small:
#                     self.small -= 1
#                     return True
#         return False


# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)


# 19 - EASY
# Input: accounts = [[1,2,3],[3,2,1]]
# Output: 6

# class Solution:
#     def maximumWealth(self, accounts: list[list[int]]) -> int:
#         m = [sum(i) for i in accounts]
#         return max(m)



# return max([sum(acc) for acc in accounts])


# 21 - EASY 
# Input: hours = [0,1,2,3,4], target = 2
# Output: 3


# class Solution:
#     def numberOfEmployeesWhoMetTarget(self, hours: list[int], target: int) -> int:
#         res = [i for i in hours if i >= target]
#         return len(res)
    

# s = Solution()
# hours, target = [0,1,2,3,4], 2
# print(s.numberOfEmployeesWhoMetTarget(hours, target))


# 22 - EASY
# Input: candies = [2,3,5,1,3], extraCandies = 3
# Output: [true,true,true,false,true] 


# class Solution:
#     def kidsWithCandies(self, candies: list[int], extraCandies: int) -> list[bool]:
#         max_candies = max(candies)
#         answer = [True if i + extraCandies >= max_candies else False for i in candies]
#         return answer





# s = Solution()
# candies = [2,3,5,1,3]
# extraCandies = 3
# print(s.kidsWithCandies(candies, extraCandies))


# class Solution:
#     def kidsWithCandies(self, candies: list[int], extraCandies: int) -> list[bool]:
#         n = len(candies)
#         maximum = max(candies)
#         answerList = []
#         for i in range(n):
#             sumValue = candies[i] + extraCandies
#             if sumValue >= maximum:
#                 answerList.append(True)
#             else:
#                 answerList.append(False)

#         return answerList
    
# s = Solution()
# candies = [2,3,5,1,3]
# extraCandies = 3
# print(s.kidsWithCandies(candies, extraCandies))


# # 20 - EASY
# # Input: str1 = "ABCABC", str2 = "ABC"
# # Output: "ABC"

# from math import gcd

# class Solution:
#     def gcdOfStrings(self, str1: str, str2: str) -> str:

#         n_str1 = len(str1)
#         n_str2 = len(str2)
#         df = gcd(n_str1, n_str2)

#         if (str1 + str2) != (str2 + str1):
#             return ""
#         else:
#             return str1[0:df]
        

# str1 = "ABC"
# str2 = "ABCABC"

# s = Solution()
# print(s.gcdOfStrings(str1, str2))


# 23 - EASY 
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]


# class Solution:
#     def moveZeroes(self, nums: list[int]) -> None:
#         nums.sort(key=bool, reverse=True)
#         return nums

# s = Solution()
# nums = [0,1,0,3,12]
# print(s.moveZeroes(nums))

# class Solution:
#     def moveZeroes(self, nums: List[int]) -> None:
#         n = len(nums)
#         j = 0
#         for i in nums:
#             if(i != 0):
#                 nums[j] = i
#                 j += 1
#         for i in range(j,n):
#             nums[i] = 0


# 24 - EASY PANDAS
# Input:
# student_data:
# [
#   [1, 15],
#   [2, 11],
#   [3, 11],
#   [4, 20]
# ]
# Output:
# +------------+-----+
# | student_id | age |
# +------------+-----+
# | 1          | 15  |
# | 2          | 11  |
# | 3          | 11  |
# | 4          | 20  |
# +------------+-----+

# import pandas as pd

# def createDataframe(student_data: list[list[int]]) -> pd.DataFrame:
#     df = pd.DataFrame(student_data, columns=['student_id', 'age'])
#     return df

# student_data = [
#   [1, 15],
#   [2, 11],
#   [3, 11],
#   [4, 20]
# ]
# pf = pd.DataFrame(createDataframe(student_data))
# print(pf)



# 25 - EASY
# Input: s = "abcd", t = "abcde"
# Output: "e"

# class Solution:
#     def findTheDifference(self, s: str, t: str) -> str:
#         pass



# 26 - EASY PANDAS



# import pandas as pd

# def getDataframeSize(players: pd.DataFrame) -> List[int]:
#     return players.shape



# 27 - EASY PANDAS


# import pandas as pd

# def selectData(df: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame(df) # whole dataframe
#     df = df[df['customer_id']==2] # dataframe with student_id==101
#     # df = df[['name','age']] # selected columns from dataframe with student_id=101
#     return df
    

# students.loc[students['student_id'] == 101, ['name', 'age']]



# 28 - EASY PANDAS


# import pandas as pd

# def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
#     df = customers.drop_duplicates(subset=['email'])
#     return df


# return customers.groupby('email').head(1)

# df = {
#     'customer_id': [1,2,3,4,5,6],
#     'name': ['Ella', 'David', 'Zachary', 'Alice', 'Finn', 'Violeta'],
#     'email': ['emily@example.com', 'michael@example.com', 'sarah@example.com', 'john@example.com', 'john@example.com', 'john@example.com']
# }



# 29 - EASY PANDAS


# import pandas as pd

# def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
#     student_without_none = students.dropna()
#     return student_without_none

    # student_without_none = students.dropna(subset='name')
    # return student_without_none


# 30 EASY PANDAS


# import pandas as pd

# def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
#     students_rename = students.rename(columns={'id': 'student_id', 'first': 'first_name', 'last': 'last_name', 'age': 'age_in_years'})
#     return students_rename



# 31 - EASY PANDAS


# import pandas as pd

# def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
#     return students.astype({'grade': int})


# 32 - EASY PANDAS


# import pandas as pd

# def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
#     products['quantity'] = products['quantity'].fillna(0)
#     return products


# 33 - EASY PANDAS


# import pandas as pd

# def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
#     """
#     Filter and sort a DataFrame of animals to find thos
#     with a weight greater than 100.

#     This function reduces memory usage by selecting only the necessary
#     columns ['name', 'weight']
#     and then filters animals based on their weight before sorting.
#     Sorting is performed on the filtered subset of heavy animals,
#     improving efficiency when dealing with large datasets.

#     Parameters:
#     - animals (pd.DataFrame): A DataFrame containing information
#     about animals, including their names and weights.

#     Returns:
#     - pd.DataFrame: A DataFrame containing the names of animals with
#     a weight greater than 100,
#       sorted by weight in descending order.
#     """
#     # Reduce the memory usage by selecting only the needed columns
#     animals = animals[["name", "weight"]]

#     # Filter animals based on their weight before sorting
#     heavy_animals = animals[animals["weight"] > 100]

#     # Sort the smaller df which is more efficient
#     heavy_animals = heavy_animals.sort_values(by="weight", ascending=False)[["name"]]

#     return heavy_animals


# 34 - EASY


# class Solution:
#     def interpret(self, command: str) -> str:
#         return command.replace("()", 'o').replace("(al)", 'al')
    
# test = "G()(al)"
# s = Solution()
# print(s.interpret(test))


# 35 - EASY


# class Solution:
#     def countPairs(self, nums: List[int], target: int) -> int:
    


# def removeDuplicates(nums: int) -> int:
#     j = 1
#     for i in range(1, len(nums)):
#         if nums[i] != nums[i - 1]:
#             nums[j] = nums[i]
#             j += 1
#     return j



# nums = [1,1,2]
# print(removeDuplicates(nums)) 



# 36 - EASY ARRAY


# class Solution:
#     def buildArray(self, nums: List[int]) -> List[int]:
#         answer = []

#         for i in range(len(nums)):
#             answer.append(nums[nums[i]])
#         return answer


# 37 - EASY ARRAY



# def mostWordsFound(sentences: list[str]) -> int:
#     # answ = []
#     new_length = 0

#     # for item in sentences:
#     #     answ.append(len(item.split()))

#     for item in sentences:
#         old_length = len(item.split())
#         if old_length > new_length:
#             new_length = old_length


#     return new_length




# sentences = ["alice and bob love leetcode", "i think so too", "this is great thanks very much"]
# print(mostWordsFound(sentences=sentences))



# 38 - EASY ARRAY


# def checkString(s: str) -> bool:
#     s2 = ''.join(sorted(s))
#     return s == s2

# s = "abab"
# print(checkString(s))


# 39 - EASY ARRAY


# class Solution:
#     def containsDuplicate(self, nums: list[int]) -> bool:
#         return True if len(set(nums)) != len(nums) else False
    


# s = Solution()
# nums = [1,2,3,1]
# print(s.containsDuplicate(nums))


# 40 - EASY 


# def runningSum(nums: list[int]):
#     for i in range(1, len(nums)):
#         nums[i] += nums[i-1]
#     return nums

# nums = [1,2,3,4]
# print(runningSum(nums))



# 41 - EASY 


# def arrayStringsAreEqual(word1: list[str], word2: list[str]) -> bool:
#     return True if ''.join(word1) == ''.join(word2) else False 



# word1 = ["ab", "c"]
# word2 = ["a", "bc"]
# print(arrayStringsAreEqual(word1, word2))



# 42 - EASY



# def countMatches(items: list[list[str]], ruleKey: str, ruleValue: str) -> int:
#     counter = 0
#     rules = {
#         'type': 0,
#         'color': 1,
#         'name': 2,
#     }

#     indx = rules[ruleKey]

#     for i in range(len(items)):
#         if items[i][indx] == ruleValue:
#             counter += 1
#     return counter

# items = [["phone","blue","pixel"],["computer","silver","lenovo"],["phone","gold","iphone"]]
# ruleKey = "color"
# ruleValue = "silver"
# print(countMatches(items, ruleKey, ruleValue))


# 43 - EASY


# def truncateSentence(s: str, k: int) -> str:
#         # s =  s.split(" ")
#         # return s[:k]
#         return " ".join(s.split(" ")[:k])

# s = "Hello how are you Contestant"
# k = 4
# print(*truncateSentence(s, k))


# 44 - EASY

# def capitalizeTitle(title: str) -> str:
#         output = list()
#         word_arr = title.split()
#         for word in word_arr:
#                 output.append(word.title()) if len(word) > 2 else output.append(word.lower())
#         return " ".join(output)
        




# title = "First leTTeR of EACH Word"
# print(capitalizeTitle(title))




# class Solution:
#     def capitalizeTitle(self, title: str) -> str:
#         ls=title.split(" ")
#         a=[]
#         for i in ls:
#             if len(i)==1 or len(i)==2: a.append(i.lower())
#             else: a.append(i.capitalize())
#         b=' '.join([str(elem) for elem in a])
#         return b
    
# s = Solution()
# title = "First leTTeR of EACH Word"
# print(s.capitalizeTitle(title))


# 45 - Easy


# def differenceOfSums(n: int, m: int) -> int:
#     # nums1 = [i for i in range(1, n + 1) if i % m != 0]
#     # nums2 = [i for i in range(1, n + 1) if i % m == 0]

#     # return sum(nums1) - sum(nums2)
#     return sum([i for i in range(1, n + 1) if i % m != 0]) - sum([i for i in range(1, n + 1) if i % m == 0])


# n = 10
# m = 3
# print(differenceOfSums(n, m))


# 46 - Easy


# def smallestEvenMultiple(n: int) -> int:
#     return n if n % 2 == 0 else n * 2


# n = 5
# print(smallestEvenMultiple(n))


# 47 - Easy


# def reverseWords(s: str) -> str:
#     ls = s.split(' ')
#     new_ls = list()

#     for i in ls:
#         new_ls.append(i[::-1])
#     result = ' '.join([str(elem) for elem in new_ls])

#     return result



# s = "Let's take LeetCode contest"
# print(reverseWords(s))



# 48 - Easy


# def maxProductDifference(nums: list[int]) -> int:
#     nums.sort()
#     a, b, c, d = nums[-1], nums[-2], nums[0], nums[1]
#     return (a * b) - (c * d)

#     # return nums[-1] * nums[-2] - nums[0] * nums[1]

# def maxProductDifference(self, nums: List[int]) -> int:
#     biggest = 0
#     second_biggest = 0
#     smallest = inf
#     second_smallest = inf
        
#     for num in nums:
#         if num > biggest:
#             second_biggest = biggest
#             biggest = num
#         else:
#             second_biggest = max(second_biggest, num)
                
#         if num < smallest:
#             second_smallest = smallest
#             smallest = num
#         else:
#             second_smallest = min(second_smallest, num)
    
#     return biggest * second_biggest - smallest * second_smallest

# nums = [5,6,2,7,4]
# print(maxProductDifference(nums))


# 48 - easy


# def isAcronym(words: list[str], s: str) -> bool:
#     # check_words = [i[0] for i in words]
#     # flag = False

#     # if len(s) == len(check_words):
#     #     for i in range(len(s)):
#     #         if s[i] == check_words[i]:
#     #             flag = True
#     #         else:
#     #             flag = False
#     #             break
#     # return flag
    
#     pref_words = [i[0] for i in words]

#     check = ''.join(pref_words)
#     return s == check

# words = ["an","apple"]
# s = "a"
# print(isAcronym(words, s))


# 49 - easy


# def uniqueMorseRepresentations(words: list[str]) -> int:
    
#     dicti = {
#         'a':".-", 'b':"-...", 'c':"-.-.", 'd':"-..", 'e':".", 'f':"..-.", 'g':"--.", 'h':"....", 'i':"..", 'j':".---", 'k':"-.-", 'l':".-..", 'm':"--", 'n':"-.", 'o':"---", 'p':".--.", 'q':"--.-", 'r':".-.", 's':"...", 't':"-", 'u':"..-", 'v':"...-", 'w':".--", 'x':"-..-", 'y':"-.--", 'z':"--.."
#         }
#     list_checkup = list()



#     for i in range(len(words)):
#         result = ''
#         check_word = words[i]

#         for j in check_word:
#             result += dicti.get(j)
#         list_checkup.append(result)
#         result = ''
    
#     return len(set(list_checkup))



# words = ["rwjje","aittjje","auyyn","lqtktn","lmjwn"]
# print(uniqueMorseRepresentations(words))


