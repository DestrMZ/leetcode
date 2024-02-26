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
# pd.DataFrame1
# +------------+---------+-----+
# | student_id | name    | age |
# +------------+---------+-----+
# | 1          | Mason   | 8   |
# | 2          | Ava     | 6   |
# | 3          | Taylor  | 15  |
# | 4          | Georgia | 17  |
# +------------+---------+-----+
# pd.DataFrame2
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

# def concatenateTables(pd.DataFrame1: pd.DataFrame, pd.DataFrame2: pd.DataFrame) -> pd.DataFrame:
#     new_pd = pd.concat([pd.DataFrame1, pd.DataFrame2])
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
#         pd.DataFrame = gcd(n_str1, n_str2)

#         if (str1 + str2) != (str2 + str1):
#             return ""
#         else:
#             return str1[0:pd.DataFrame]
        

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
#     pd.DataFrame = pd.DataFrame(student_data, columns=['student_id', 'age'])
#     return pd.DataFrame

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

# def selectData(pd.DataFrame: pd.DataFrame) -> pd.DataFrame:
#     pd.DataFrame = pd.DataFrame(pd.DataFrame) # whole dataframe
#     pd.DataFrame = pd.DataFrame[pd.DataFrame['customer_id']==2] # dataframe with student_id==101
#     # pd.DataFrame = pd.DataFrame[['name','age']] # selected columns from dataframe with student_id=101
#     return pd.DataFrame
    

# students.loc[students['student_id'] == 101, ['name', 'age']]



# 28 - EASY PANDAS


# import pandas as pd

# def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
#     pd.DataFrame = customers.drop_duplicates(subset=['email'])
#     return pd.DataFrame


# return customers.groupby('email').head(1)

# pd.DataFrame = {
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

#     # Sort the smaller pd.DataFrame which is more efficient
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


# 50 - easy


# def getDecimalValue(head: list) -> int:
#     new_head = ''
#     for i in head:
#         new_head += str(i)
#     return int(new_head, 2) * 2




# head = [1,0,1]
# print(getDecimalValue(head))


# 51 - easy

# first option 
# def numOfStrings(patterns: list[str], word: str) -> int:
#     counter = 0
    
#     for i in range(len(patterns)):
#         if patterns[i] in word:
#             counter += 1
#     return counter

# second 

# def numOfStrings(patterns: list[str], word: str) -> int:
#     return sum(patterns in word for patterns in word)


# patterns = ["a","b","c"]
# word = "aaaaabbbbb"
# print(numOfStrings(patterns, word))


# 52 - easy


# def reversePrefix(word: str, ch: str) -> str:
#     index = word.find(ch) + 1
#     revers_word = word[:index]
#     revers = revers_word[::-1]
#     return revers + word[index:]

# word = "abcdefd"
# ch = "d"

# print(reversePrefix(word, ch))


# 53 - easy

# import pandas as pd

# def createDataframe(student_data: list[list[int]]) -> pd.DataFrame:
#   return pd.DataFrame(student_data, columns = ["student_id", "age"], index = range(1, len(student_data) + 1))
  



# student_data = [
#   [1, 15],
#   [2, 11],
#   [3, 11],
#   [4, 20]
# ]

# print(createDataframe(student_data))


# 54 - easy


# def sortPeople(names: list[str], heights: list[int]) -> list[str]:
#   piople = sorted(list(zip(names, heights)), key=lambda item: item[1], reverse=True)
#   return [name for name, heights in piople]



# names = ["Mary","John","Emma"]
# heights = [180,165,170]

# print(sortPeople(names, heights))


# 55 - easy
   

# def fizzBuzz(n: int) -> list[str]:
#     check_list = []

#     for i in range(1, n + 1):
#         if i % 15 == 0: check_list.append('FizzBuzz')
#         elif i % 3 == 0: check_list.append('Fizz')
#         elif i % 5 == 0: check_list.append('Buzz')
#         else: check_list.append(str(i))

#     return check_list

#     # return ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i) for i in range(1, n + 1)]

# class Solution:
#     def fizzBuzz(self, n: int):
#         ans = []
#         for i in range(1, n + 1):
#             ans.append(
#                 "FizzBuzz" if i % 15 == 0 else
#                 "Buzz" if i % 5 == 0 else
#                 "Fizz" if i % 3 == 0 else
#                 str(i)
#             )
#         return ans
    

# n = 3

# print(fizzBuzz(n))


# 56 - easy


# def numberOfSteps(num: int) -> int:
#     step = 0

#     while num != 0:
#         step += 1
#         if num % 2 == 0:
#             num /= 2
#         else:
#             num -= 1
#     return step


# num = 14
# print(numberOfSteps(num))


# 57 - easy

# from collections import Counter

# def canConstruct(ransomNote: str, magazine: str) -> bool:
#     cnt = Counter([magazine])


# ransomNote = "aa"
# magazine = "ab"
# print(canConstruct(ransomNote, magazine))


#### Pandas

# import pandas as pd

# def big_countries(world: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame(world)
#     cont = df[(df['area'] >= 3000000) | (df['population'] >= 25000000)][['name', 'population', 'area']]
#     return cont



# data = [
#     ['Afghanistan', 'Asia', 652230, 25500100, 20343000000],
#     ['Albania', 'Europe', 28748, 2831741, 12960000000], 
#     ['Algeria', 'Africa', 2381741, 37100000, 188681000000], 
#     ['Andorra', 'Europe', 468, 78115, 3712000000], 
#     ['Angola', 'Africa', 1246700, 20609294, 100990000000],
#     ]

# world = pd.DataFrame(data, columns=['name', 'continent', 'area', 'population', 'gdp']).astype({'name':'object', 'continent':'object', 'area':'Int64', 'population':'Int64', 'gdp':'Int64'})


# print(big_countries(world))


### Pandas



# import pandas as pd

# def find_products(products: pd.DataFrame) -> pd.DataFrame:
#     cont = products[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y')][['product_id']]
#     return cont

# data = [
#     ['0', 'Y', 'N'], 
#     ['1', 'Y', 'Y'], 
#     ['2', 'N', 'Y'], 
#     ['3', 'Y', 'Y'], 
#     ['4', 'N', 'N']
#     ]

# products = pd.DataFrame(data, columns=['product_id', 'low_fats', 'recyclable']).astype({'product_id':'int64', 'low_fats':'category', 'recyclable':'category'})
# print(find_products(products))



### easy


# class Solution:
#     def sortColors(self, nums: list[int]) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         low, mid, high = 0, 0, len(nums) - 1
            
#         while mid <= high:
#             if nums[mid] == 0:
#                 nums[low], nums[mid] = nums[mid], nums[low]
#                 low += 1
#                 mid += 1
#             elif nums[mid] == 1:
#                 mid += 1
#             else:
#                 nums[mid], nums[high] = nums[high], nums[mid]
#                 high -= 1


### easy 


# def isAnagram(s: str, t: str) -> bool:
    # return sorted(s) == sorted(t)


### medium 


# class Solution:
#     def isAdditiveNumber(self, num: str) -> bool:
        
#         if len(num) < 3:   ## Делаем проверку, на колличество проверяемого массива
#             return False
        

## easy

# class Solution:
#     def shuffle(self, nums: list[int], n: int) -> list[int]: 
#         ls = list()

#         for i in range(n):
#             ls.append(nums[i])
#             ls.append(nums[i + n])
#         return ls

# nums = [1,2,3,4,4,3,2,1]
# n = 4
# s = Solution()
# print(s.shuffle(nums, n))

        

## easy


# class Solution:
#     def countPairs(self, nums: list[int], target: int) -> int:
#         counter = 0

#         for i in range(len(nums) - 1):
#             for j in range(1, len(nums)):

#                 if (i < j) and (nums[i] + nums[j]) < target:
#                     counter += 1

#         return counter
                


# nums = [-1,1,2,3,1]
# target = 2
# s = Solution()
# print(s.countPairs(nums, target))


### вариант с двумя указателями


# class Solution:
#     def countPairs(self,nums:list[int], target:int) -> int:
#         nums.sort()

#         counter = 0

#         left, right = 0, len(nums) - 1
#         while left < right:
#             if nums[left] + nums[right] < target:
#                 counter += right - left
#                 left += 1
#             else:
#                 right -= 1

#         return counter
    

# nums = [-1,1,2,3,1]
# target = 2
# s = Solution()
# print(s.countPairs(nums, target))


# class Solution:
#     def countPairs(self, nums: list[int], target: int) -> int:
#         nums.sort()
#         counter = 0
#         left, right = 0, len(nums) - 1
#         while left < right:
#             if nums[left] + nums[right] < target:
#                 counter += right - left
#                 left += 1
#             else:
#                 right -= 1
#         return counter


# nums = [-1,1,2,3,1]
# target = 2
# s = Solution()
# print(s.countPairs(nums, target))



## easy


# class Solution:
#     def smallerNumbersThanCurrent(self, nums: list[int]) -> list[int]:
#         smaller_list = []
#         counter = 0

#         for i in range(len(nums)):
#             for j in range(len(nums)):
#                 if (j != i) and (nums[j] < nums[i]):
#                     counter += 1
#             smaller_list.append(counter)
#             counter = 0

#         return smaller_list


# class Solution:
#     def smallerNumbersThanCurrent(self, nums: list[int]) -> list[int]:
#         sorted_unique_nums = sorted(set(nums))  # уникальные числа, отсортированные
#         smaller_counts = {num: i for i, num in enumerate(sorted_unique_nums)}
#         return [smaller_counts[num] for num in nums]




# s = Solution()
# nums = [8,1,2,2,3]
# print(s.smallerNumbersThanCurrent(nums))



### easy


# class Solution:
#     def findMaxConsecutiveOnes(self, nums: list[int]) -> int:
#         counter = 0
#         ls = []
#         for i in range(len(nums)):
#             if nums[i] == 1:
#                 counter += 1
#             else:
#                 ls.append(counter)
#                 counter = 0
#         ls.append(counter)
#         return max(ls)
    

# s = Solution()
# nums = [1, 0, 0, 1, 1, 1, 0]
# print(s.findMaxConsecutiveOnes(nums))



### easy


# class Solution:
#     # def findNumbers(self, nums: list[int]) -> int:
#         # counter = 0
#         # for i in nums:
#         #     if len(str(i)) % 2 == 0:
#         #         counter += 1
#         # return counter
    

#         def findNumbers(self, nums: list[int]) -> int:
#             def digit_count(number):
#                 count = 0
#                 while number > 0:
#                     count += 1
#                     number //= 10
#                 return count
            
#             return sum(1 for i in nums if digit_count(i) % 2 == 0)


# s = Solution()
# nums = [555,901,482,1771]
# print(s.findNumbers(nums))


### easy


# class Solution:
#     def sortedSquares(self, nums: list[int]) -> list[int]:

#         for i in range(len(nums)):
#             nums[i] *= nums[i]
#         nums.sort()
#         return nums



# s = Solution()
# nums = [-4,-1,0,3,10]
# print(s.sortedSquares(nums))


### easy 

# from collections import Counter


# def firstUniqChar(s: str) -> int:
#     r_split = [s for s in s]
#     my_dict = Counter(r_split)

#     for i in range(len(r_split)):
#         if my_dict[r_split[i]] == 1:    
#             return i
#     else:
#         return -1
#     # return my_dict


# s = "loveleetcode"
# print(firstUniqChar(s))


# def firstUniqChar(s: str) -> int:
#     my_dict = Counter(s)

#     for i in range(len(s)):
#         if my_dict[s[i]] == 1:
#             return i
#     else:
#         return -1
# s = "loveleetcode"
# print(firstUniqChar(s))


# s = "loveleetcode"
# mp = {}

# for a in s:
#     mp[a] = mp.get(a, 0) + 1

# print(mp)



# from collections import defaultdict

# class Solution:
#     def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
#         my_dict = defaultdict(list)

#         for word in strs:
#             sort_word = tuple(sorted(word))
#             my_dict[sort_word].append(word)
#         return list(my_dict.values())




# s = Solution()
# strs = ["eat","tea","tan","ate","nat","bat"]
# print(s.groupAnagrams(strs))



# class Solution:
#     def convertTemperature(self, celsius: float) -> list[float]:
#         return [celsius + 273.15, celsius * 1.80 + 32.00]



# s = Solution()
# celsius = 36.50
# print(s.convertTemperature(celsius))




# easy


# def manacher_algorithm(s):
#     s_trans = '@#' + '#'.join(list(s)) + '#$'
#     print(s_trans)


#     p = [0] * len(s)
#     c = 0
#     r = 0

#     for i in range(1, len(s_trans) - 1):
#         mirr = 2 * c - i
#         print(mirr, 'mirr')

#         if i < r:
#             p[i] = min(r - i, p[mirr])

#         while s_trans[i + (1 + p[i])] == s_trans[i - (1 + p[i])]:
#             p[i] += 1
        
#         if i + p[i] > r:
#             c = i
#             r = i + p[i]

#         counter_polindromes = 0
#         for lenght in p:
#             counter_polindromes += (lenght // 2)

#         return counter_polindromes
    

# s = 'aaa'
# print(manacher_algorithm(s))



# def manacher_algorithm(s):
#     s_transformed = '@#' + '#'.join(s) + '#$'

#     p = [0] * len(s_transformed)
#     c = 0 
#     r = 0 

#     for i in range(1, len(s_transformed) - 1):
#         mirr = 2 * c - i
        
#         if i < r:
#             p[i] = min(r - i, p[mirr])
        
#         while s_transformed[i + (p[i] + 1)] == s_transformed[i - (p[i] + 1)]:
#             p[i] += 1
        
#         if i + p[i] > r:
#             c = i
#             r = i + p[i]

#     count = 0
#     for length in p:
#         count += (length + 1) // 2
#     return count


# s = "aaa"
# print(manacher_algorithm(s))




# easy


# class Solution:
#     def removeOuterParentheses(self, s: str) -> str:
#         stack = []
#         brackets_map = {
#             ')': '(',
#             ']': '[', 
#             '}': '{',
#             }

#         for i, char in enumerate(s, start=1):
#             if char in brackets_map.values():
#                 stack.append((char, i))
#             elif char in brackets_map:
#                 if not stack or stack[-1][0] != brackets_map[char]:
#                     return i
#                 stack.pop()

#         if stack:
#             return[0][1]
    
#         return 'Success'





# parantheses = '([](){([])})'
# s = Solution()
# print(s.removeOuterParentheses(parantheses))





# def removeOuterParentheses(s: str) -> str:
# parantheses = input()
# stack = []
# index = -1
# result = True
# brackets_map = {
#     ')': '(',
#     ']': '[', 
#     '}': '{',
#     }


# for i, char in enumerate(parantheses, start=1):
#     if char in brackets_map.values():
#         stack.append((char, i))
#     elif char in brackets_map:
#         if not stack or stack[-1][0] != brackets_map[char]:
#             result = False
#             index = i
#             break
#         stack.pop()

# if stack and result:
#     result = False
#     indes = stack[0][1]

# if result:
#     print('Success')
# else:
#     print(index)




# def check_brackets(s: str) -> str:
#     stack = []
#     brackets_map = {
#         ')': '(',
#         ']': '[', 
#         '}': '{',
#         }

#     for i, char in enumerate(s, start=1):
#         if char in brackets_map.values():
#             stack.append((char, i))
#         elif char in brackets_map:
#             if not stack or stack[-1][0] != brackets_map[char]:
#                 return i
#             stack.pop()

#     if stack:
#         return[0][1]
    
#     return 'Success'





# parantheses = input()
# print(check_brackets(parantheses))




# def run(string):
#     braces = {')': '(', '}': '{', ']': '['}
#     stack = []
#     for i, c in enumerate(string, start=1):
#         if c in braces.values():
#             stack.append((c, i))
#         if c in braces and (not stack or braces[c] != stack.pop()[0]):
#             return i
#     return stack.pop()[1] if stack else 'Success'


# if __name__ == '__main__':
#     print(run(input()))





# def find_Majority(nums: list[int]) -> int:
#     counter = 0
#     candidates = None

#     for num in nums:
#         if counter == 0:
#             candidates = num
        
#         counter += (1 if num == candidates else -1)
    
#     return candidates



# easy 



# class Solution:
#     def firstPalindrome(self, words: list[str]) -> str:
#         for i in words:
#             if i == i[::-1]:
#                 return i
#         return ''



# words = ["notapalindrome","racecar"]
# s = Solution()
# print(s.firstPalindrome(words))


# easy

# from collections import Counter

# class Solution:
#     def topKFrequent(self, nums: list[int], k: int) -> list[int]:
#         pass




# s = Solution()
# nums = [1,1,1,2,2,3]
# k = 2
# print(s.topKFrequent(nums, k))




# class Solution:
#     def maxProfit(self, prices: list[int]) -> int:
#         min_price, max_profit = prices[0], 0

#         for i in range(1, len(prices)):
#             if prices[i] < min_price:
#                 min_price = prices[i]
            
#             if prices[i] - min_price > max_profit:
#                 max_profit = prices[i] - min_price
            
#         return max_profit



# prices = [7,6,4,3,1]
# s = Solution()
# print(s.maxProfit(prices))




# class Solution:
#     def isPalindrome(self, s: str) -> bool:
#         string = ''.join(i for i in s.lower() if i.isalnum())
#         return string == string[::-1]
    
#         # for i in s.lower():
#         #     if i in 'abcdefghijklmnopqrstuvwxyz0123456789':
#         #         string += i

#         # return string == string[::-1]




# res = "0P"
# s = Solution()
# print(s.isPalindrome(res))





# class Solution:
#     def search(self, nums: list[int], target: int) -> int:
#         return nums.index(target) if target in nums else -1




# s = Solution()
# nums = [-1,0,3,5,9,12]
# target = 9
# print(s.search(nums, target))