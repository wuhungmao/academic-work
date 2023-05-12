package boggle;

import java.lang.reflect.Array;
import java.util.*;

/**
 * The BoggleGrid class for the first Assignment in CSC207, Fall 2022
 * The BoggleGrid represents the grid on which we play Boggle 
 */
public class BoggleGrid {

    /**
     * size of grid
     */  
    private final int size;
    /**
     * characters assigned to grid
     */      
    final char[][] board;

    private final String dice = "AAAFRSAAEEEEAAFIRSADENNNAEEEEMAEEGMUAEGMNNAFIRSYBJKQXZCCNSTWCEIILTCEILPTCEIPSTDDLNOR" +
            "DDHNOTDHHLORDHLNOREIIITTEMOTTTENSSSUFIPRSYGORRVWHIPRRYNOOTUWOOOTTU";

    /** BoggleGrid constructor
     * ----------------------
     * @param size  The size of the Boggle grid to initialize
     */
    public BoggleGrid(int size) {
        this.size = size;
        this.board = new char[size][size];
        Random r = new Random();
        for (int row = 0; row < this.size; row++) { //Assign random letters from dice to the board
            for (int col = 0; col < this.size; col++) {
                this.board[row][col] = dice.charAt(r.nextInt(150));
            }
        }
    }

    /**
     * Provide a nice-looking string representation of the grid,
     * so that the user can easily scan it for words.
     *
     * @return String to print
     */
    @Override
    public String toString() {
        String boardString = "";
        for(int row = 0; row < this.size; row++){
            for(int col = 0; col < this.size; col++){
                boardString += this.board[row][col] + " ";
            }
            boardString += "\n";
        }
        return boardString;
    }

    /**
     * @return int the number of rows on the board
     */
    public int numRows() {
        return this.size;
    }

    /**
     * @return int the number of columns on the board (assumes square grid)
     */
    public int numCols() {
        return this.size;
    }

    /**
     * @return char the character at a given grid position
     */
    public char getCharAt(int row, int col) {
        return this.board[row][col];
    }

    /**
     * Replace the letter with character ch at the given row and col.
     */
    public void replace(int row, int col, char ch) {
        this.board[row][col] = ch;
    }
}
