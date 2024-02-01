
# # init stack
# class Stack():
#     def __init__(self):
#         self.items = []

#     def is_empty(self):
#         return len(self.items) == 0

#     def push(self, items):
#         return self.items.append(items)

#     def pop(self):
#         if not self.is_empty():
#             return self.items.pop()
        
#         else:
#             return 'Stack empty'
        
#     def peek(self):
#         if not self.is_empty():
#             return self.items[-1]

#         else:
#             return 'Stack empty'
        
#     def size(self):
#         return len(self.items)
    


# my_stack = Stack()

# my_stack.push('Ivan')
# my_stack.push('Sasha')
# my_stack.push('Alex')

# print(my_stack.size())
# print(my_stack.items)


# my_stack.pop()
# my_stack.peek()

# print(my_stack.items)

# my_stack.pop()

# print(my_stack.items)
# print(my_stack.size())





### Linked list


# class NodeList:
#     def __init__(self, data):
#         self.data = data
#         self.next = None



# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def append(self, data):
#         new_node = NodeList(data)
#         if self.head is None:
#             self.head = new_node
#             return new_node
        
#         last_node = self.head
#         while last_node.next:
#             last_node = last_node.next
#         last_node.next = new_node

#     def print_list(self):
#         cur_node = self.head
#         while cur_node:
#             print(cur_node.data)
#             cur_node = cur_node.next

#     def find_middle_to_end(self):
#         slow = self.head
#         fast = self.head

#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

#         while slow:
#             print(slow.data)
#             slow = slow.next
            
#         return slow

# linked_list = LinkedList()

# linked_list.append(1)
# linked_list.append(2)
# linked_list.append(3)
# linked_list.append(4)
# linked_list.append(5)

# linked_list.find_middle_to_end()
# # linked_list.print_list()







# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# class Solution:
#     def middleNode(self, head: ListNode):
#         slow = head
#         fast = head

#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

#         return slow
    

# head = [1,2,3,4,5]
# solution = Solution()
# middle_note = solution.middleNode(head)



class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def middleNode(self, head: ListNode):
        first_ptr = head
        second_ptr = head
        while first_ptr and first_ptr.next:
            second_ptr = second_ptr.next
            first_ptr = first_ptr.next.next
        return second_ptr
    


head = [1,2,3,4,5]
solution = Solution()

print(solution.middleNode(head))