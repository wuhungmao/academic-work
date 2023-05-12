package test;

import java.util.*;

import boggle.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class BoggleModelTests {

    //BoggleModel inputWord Test
    @Test
    void inputWord() {
        BoggleModel model = new BoggleModel("PVP", 9, "Normal");
        model.inputWord("AB");
        model.inputWord("ABC");
        model.inputWord("abc");
        Set<String> expected = new HashSet<>(List.of("ABC"));
        assertEquals(expected, model.game.getPlayerWords(1));
    }

    //BoggleModel removeWord Test
    @Test
    void removeWord() {
        BoggleModel model = new BoggleModel("PVP", 9, "Normal");
        model.inputWord("ABC");
        model.removeWord("ABC");
        Set<String> expected = new HashSet<>();
        assertEquals(expected, model.game.getPlayerWords(1));
    }

    //BoggleModel next Test
    @Test
    void next() {
        BoggleModel model = new BoggleModel("PVP", 9, "Normal");
        model.next();
        model.inputWord("ABC");
        Set<String> expected = new HashSet<>(List.of("ABC"));
        assertEquals(expected, model.game.getPlayerWords(2));
    }

    //BoggleModel nextRound Test
    @Test
    void nextRound() {
        BoggleModel model = new BoggleModel("PVP", 9, "Normal");
        model.inputWord("ABC");
        model.next();
        Set<String> expected = new HashSet<>(List.of("ABC"));
        assertEquals(expected, model.game.getPlayerWords(1));
    }

}
