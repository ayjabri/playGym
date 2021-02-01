"""
Connect 4 Game
"""
import random
from lib.utils import Experience
from lib.cylib import functions_cy as cy

class Connect4(object):
    def __init__(self, rows=6, cols=7, bits_in_len=3, count_to_win=4):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.space = self.rows*self.cols
        self.bits_in_len = bits_in_len
        self.count_to_win = count_to_win
        self.white_player = 0
        self.black_player = 1
        self.init_state_list = [[] for _ in range(self.cols)]
        self.init_state = cy.encode_lists(self.init_state_list)
        self.reset()

    def reset(self):
        self.cur_state = None
        self.children = {}
        self.steps = 0

    def expand_children(self, state_int, player):
        children = []
        valid_moves = self.possible_moves(state_int)
        for move in valid_moves:
            child, _ = self.move(state_int, move, player)
            children.append((move, child))
        return children

    # def bits_to_int(self, bits):
    #     res = 0
    #     for b in bits:
    #         res *= 2
    #         res += b
    #     return res

    # def int_to_bits(self, num, bits):
    #     res = []
    #     for _ in range(bits):
    #         res.append(num % 2)
    #         num //= 2
    #     return res[::-1]

    # def encode_lists(self, field_lists):
    #     """
    #     Encode lists representation into the binary numbers
    #     :param field_lists: list of GAME_COLS lists with 0s and 1s
    #     :return: integer number with encoded game state
    #     """
    #     assert isinstance(field_lists, list)
    #     assert len(field_lists) == self.cols

    #     bits = []
    #     len_bits = []
    #     for col in field_lists:
    #         bits.extend(col)
    #         free_len = self.rows-len(col)
    #         bits.extend([0] * free_len)
    #         len_bits.extend(self.int_to_bits(free_len, bits=self.bits_in_len))
    #     bits.extend(len_bits)
    #     return self.bits_to_int(bits)

    # def decode_binary(self, state_int):
    #     """
    #     Decode binary representation into the list view
    #     :param state_int: integer representing the field
    #     :return: list of GAME_COLS lists
    #     """
    #     assert isinstance(state_int, int)
    #     bits = self.int_to_bits(
    #         state_int, bits=self.space + self.cols*self.bits_in_len)
    #     res = []
    #     len_bits = bits[self.space:]
    #     for col in range(self.cols):
    #         vals = bits[col*self.rows:(col+1)*self.rows]
    #         lens = self.bits_to_int(
    #             len_bits[col*self.bits_in_len:(col+1)*self.bits_in_len])
    #         if lens > 0:
    #             vals = vals[:-lens]
    #         res.append(vals)
    #     return res

    # def _binary_to_array(self, state_int):
    #     rep = np.zeros((2, self.cols, self.rows))
    #     bits = cy.int_to_bits(
    #         state_int, bits=self.space + self.cols*self.bits_in_len)
    #     len_bits = bits[self.space:]
    #     for col in range(self.cols):
    #         vals = bits[col*self.rows:(col+1)*self.rows]
    #         lens = cy.bits_to_int(
    #             len_bits[col*self.bits_in_len:(col+1)*self.bits_in_len])
    #         if lens > 0:
    #             vals = vals[:-lens]
    #             for i, cell in enumerate(vals):
    #                 if cell == 0:
    #                     rep[0, col, i] += 1
    #                 else:
    #                     rep[1, col, i] += 1
    #     return rep

    def possible_moves(self, state_int):
        """
        This function could be calculated directly from bits, but I'm too lazy
        :param state_int: field representation
        :return: the list of columns which we can make a move
        """
        assert isinstance(state_int, int)
        field = cy.decode_binary(state_int)
        return [idx for idx, col in enumerate(field) if len(col) < self.rows]

    def _check_won(self, field, col, delta_row):
        """
        Check for horisontal/diagonal win condition for the last player moved in the column
        :param field: list of lists
        :param col: column index
        :param delta_row: if 0, checks for horisonal won, 1 for rising diagonal, -1 for falling
        :return: True if won, False if not
        """
        player = field[col][-1]
        coord = len(field[col])-1
        total = 1
        # negative dir
        cur_coord = coord - delta_row
        for c in range(col-1, -1, -1):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord -= delta_row
        # positive dir
        cur_coord = coord + delta_row
        for c in range(col+1, self.cols):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord += delta_row
        return False

    def is_draw(self, state_int):
        valid_moves = self.possible_moves(state_int)
        return valid_moves == 0

    def move(self, state_int, col, player):
        """
        Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
        :param state_int: current state
        :param col: column to make a move
        :param player: player index (PLAYER_WHITE or PLAYER_BLACK
        :return: tuple of (state_new, won). Value won is bool, True if this move lead
        to victory or False otherwise (but it could be a draw)
        """
        assert isinstance(state_int, int)
        assert isinstance(col, int)
        assert 0 <= col < self.cols
        assert player == self.black_player or player == self.white_player
        field = cy.decode_binary(state_int)
        assert len(field[col]) < self.rows
        field[col].append(player)
        # check for victory: the simplest vertical case
        suff = field[col][-self.count_to_win:]
        won = suff == [player] * self.count_to_win
        if not won:
            won = self._check_won(field, col, 0) or self._check_won(
                field, col, 1) or self._check_won(field, col, -1)
        state_new = cy.encode_lists(field)
        return state_new, won

    def render(self, state_int):
        state_list = cy.decode_binary(state_int)
        data = [['_'] * self.cols for _ in range(self.rows)]
        for col_idx, col in enumerate(state_list):
            for rev_row_idx, cell in enumerate(col):
                row_idx = self.rows - rev_row_idx - 1
                data[row_idx][col_idx] = str(cell)
        return data

    def player_0_random(self):
        game_history = []
        cur_state = self.init_state
        cur_player = random.choice([self.white_player, self.black_player])
        won = False
        step = 0
        valid_moves = self.possible_moves(cur_state)
        while True:
            step += 1
            action = random.choice(valid_moves)
            cur_state, won = self.move(cur_state, action, cur_player)
            game_history.append(Experience(cur_state, action, cur_player))
            if won:
                value = 1. if cur_player==0 else -1.
                print(f'player {cur_player} won in {step} steps')
                break
            valid_moves = self.possible_moves(cur_state)
            if len(valid_moves) == 0:
                value = 0.
                print('Draw')
                break
            cur_player = 1 - cur_player
        game_history.append(value)
        return game_history

    def print_board(self, state_int):
        state = self.render(state_int)
        print('='*(1+self.rows*2))
        print(' '.join(str(x) for x in range(self.cols)))
        for y in range(self.rows):
            print(' '.join(state[y][x] for x in range(self.cols)))


game = Connect4()
