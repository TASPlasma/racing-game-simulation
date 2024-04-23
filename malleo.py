import math

class Malleo:
    """
    Given N bags and K slots,
    put i bags in i mod kth slot
    Stop putting in bags if there is not enough to put in
    ith slot
    Note that some remainder may be left over
    """
    def __init__(self, N, K):
        self.N = N
        self.K = K

    def num_passes(self):
        """
        A pass is defined to be a full cycle through
        all K slots
        This maybe returns the number of full passes???
        """
        a = 2 * self.K^2
        b = -self.K^2 + self.K
        c = -2 * self.N
        rt_disc = math.sqrt(b ** 2 - 4*a*c)
        outputs = [math.floor((-b + (2*i - 1) * rt_disc) / (2 * a)) for i in range(2)]

        return outputs
    
    def solve(self):
        total = self.N # total remaining bags
        something = True
        output = [0 for i in range(self.K)]
        for i in range(self.N):
            bags_ith_slot = i % self.K
            if total < bags_ith_slot:
                break
            else:
                output[i] += bags_ith_slot
                total -= bags_ith_slot

        return output
    
# class TestCase:
#     def __init__(self, N, K, known_arr):
#         self.N = N
#         self.K = K
#         self.known_arr = known_arr
    
#     def test_case(self):
#         sol = Malleo(self.N, self.K)
#         return f'Expected: {self.known_arr}, got: {sol.solve()}'
    
# ex1 = TestCase(23, 5, [7, 2, 3, 4, 5])
# print(ex1.test_case())
    
my_obj = Malleo(23, 5)
print(f"Expected: {[7, 2, 3, 4, 5]}, got: {my_obj.solve()}")


        



