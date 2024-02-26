# # Алгоритм "большинства голосов Бойера - Мура"
# # Подходит для поиска наиболее встречаемого элемента в массиве


# def majorityElement(nums: list[int]) -> int:
#     counter = 0
#     candidate = -1

#     for i in range(len(nums)):
#         if counter == 0:
#             candidate = nums[i]
#             counter = 1
#         else:
#             if nums[i] == candidate:
#                 counter += 1
#             else:
#                 counter -= 1

#     counter = 0
    
#     for i in range(len(nums)):
#         if nums[i] == candidate:
#             counter += 1
    
#     if counter > len(nums) // 2:
#         return candidate
 

# nums = [1,1,1,1,2,3,5]
# print(majorityElement(nums))


# #### АЛЬТЕРНАТИВНЫЙ ВАРИАНТ ЭТОГО АЛГОРИТМА
# ## реализованный нейросетью


# def find_Majority(nums: list[int]) -> int:
#     counter = 0
#     candidates = None

#     for num in nums:
#         if counter == 0:
#             candidates = num
        
#         counter += (1 if num == candidates else -1)
    
#     return candidates