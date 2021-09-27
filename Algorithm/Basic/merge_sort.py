def merge_sort(arr):

    # base case
    if len(arr) < 2:
        return arr

    # recursion
    mid = len(arr)//2
    left = arr[:mid]
    right = arr[mid:]
    merge_sort(left)
    merge_sort(right)

    # merge
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arr[k] = left[i]
            i = i + 1
        else:
            arr[k] = right[j]
            j = j + 1
        k = k + 1

    while i < len(left):
        arr[k] = left[i]
        i = i + 1
        k = k + 1

    while j < len(right):
        arr[k] = right[j]
        j = j + 1
        k = k + 1


if __name__ == "__main__":
    arr = [9, 3, 19, 20, 100, 3, 45]
    merge_sort(arr)
    print(arr)

