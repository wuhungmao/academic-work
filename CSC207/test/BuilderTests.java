package test;

import Builder.ConcreteGameBuilders.HOLEBuilder;
import Builder.ConcreteGameBuilders.PVEBuilder;
import Builder.ConcreteGameBuilders.PVPBuilder;

import boggle.BoggleModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class BuilderTests {


    @Test
    void testHOLEGameModeCreation() {
        HOLEBuilder Hbuilder = new HOLEBuilder();
        Hbuilder.setSize(8);

        BoggleModel model = Hbuilder.createGame();

        char[][] grid = model.game.getGrid();

        // Make sure there are n = grid size, ((n * n) * 0.75) (truncated) letters in the game!
        int letter_count = 0;
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (grid[row][col] != ' ') {
                    letter_count += 1;
                }
            }
        }

        assertEquals(48, letter_count);
        assertEquals("PVE: HOLE", model.gameMode);
        assertEquals("Easy", model.getLevel());
    }

    @Test
    void testPVEGameModeCreation() {
        PVEBuilder CBuilder = new PVEBuilder();
        CBuilder.setDifficulty("Hard");
        CBuilder.setSize(10);

        BoggleModel model = CBuilder.createGame();

        char[][] grid = model.game.getGrid();

        int sizeCheck = 0;
        for (int row = 0; row < 10; row++) {
            for (int col = 0; col < 10; col++) {
                if (grid[row][col] != ' ') {
                    sizeCheck += 1;
                }
            }
        }

        assertEquals(100, sizeCheck);
        assertEquals("PVE", model.gameMode);
        assertEquals("Hard", model.getLevel());
    }

    @Test
    void testPVPGameModeCreation() {
        PVPBuilder PBuilder = new PVPBuilder();
        PBuilder.setSize(5);

        BoggleModel model = PBuilder.createGame();

        char[][] grid = model.game.getGrid();

        int sizeCheck = 0;
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 5; col++) {
                if (grid[row][col] != ' ') {
                    sizeCheck += 1;
                }
            }
        }

        assertEquals(25, sizeCheck);
        assertEquals("PVP", model.gameMode);
        assertEquals("None", model.getLevel());
    }
}
