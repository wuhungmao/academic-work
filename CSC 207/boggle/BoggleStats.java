package boggle;

import com.sun.javafx.collections.ObservableSetWrapper;
import javafx.collections.ObservableList;
import javafx.collections.ObservableListBase;
import javafx.collections.ObservableSet;

import java.util.*;

/**
 * The BoggleStats will contain statistics related to game play Boggle
 */
public class BoggleStats {

    /**
     * set of words player1 finds in a given round
     */
    Set<String> p1Words;

    /**
     * set of words the player2 finds in a given round
     */
    Set<String> p2Words;
    /**
     * player1's score for the current round
     */
    private int p1Score;
    /**
     * player2's score for the current round
     */
    int p2Score;
    /**
     * player1's total score across every round
     */
    private int p1ScoreTotal;
    /**
     * player2's total score across every round
     */
    private int p2ScoreTotal;
    /**
     * the average number of words, per round, found by the player
     */
    private double p1AverageWords;
    /**
     * the average number of words, per round, found by the computer
     */
    private double p2AverageWords;
    /**
     * the current round being played
     */
    protected int round;

    /** BoggleStats constructor
     * ----------------------
     * Sets round, totals and averages to 0.
     * Initializes word lists (which are sets) for computer and human players.
     */
    BoggleStats() {
        round = 0;
        p1ScoreTotal = 0;
        p1AverageWords = 0;
        p2ScoreTotal = 0;
        p2AverageWords = 0;
        p1Words  = new HashSet<>();
        p2Words  = new HashSet<>();
    }

    /**
     * End a given round.
     * This will clear out the human and computer word lists, so we can begin again.
     * The resets the current scores for each player to zero.
     */
    void endRound() {
        p2Words.clear();
        p2Score = 0;
        p1Words.clear();
        p1Score = 0;
    }

    /**
     * Summarize one round of boggle.
     * Calculate the scores;
     * @return Player information as a hashmap with 1, 2 as keys (p1, p2)
     */
    HashMap<Integer, String> summarizeRound() {
        for (String word : p1Words) p1Score += word.length() - 2;
        for (String word : p2Words) p2Score += word.length() - 2;
        round += 1;
        p1ScoreTotal += p1Score;
        p1AverageWords = (p1AverageWords * (round - 1) + p1Words.size()) / (round);
        p2ScoreTotal += p2Score;
        p2AverageWords = (p2AverageWords * (round - 1) + p2Words.size()) / (round);
        HashMap<Integer, String> summary = new HashMap<>();
        summary.put(0, String.valueOf(round));
        summary.put(1, "Score in this round: " + p1Score + "\nTotal score: " + p1ScoreTotal + "\nAverage words found per round: " + p1AverageWords);
        summary.put(2, "Score in this round: " + p2Score + "\nTotal score: " + p2ScoreTotal + "\nAverage words found per round: " + p2AverageWords);
        return summary;
    }

}