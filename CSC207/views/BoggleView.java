package views;

import Builder.ConcreteGameBuilders.GameBuilder;
import Builder.ConcreteGameBuilders.HOLEBuilder;
import Builder.ConcreteGameBuilders.PVEBuilder;
import Builder.ConcreteGameBuilders.PVPBuilder;
import boggle.BoggleModel;
import extraEventHandler.BoggleEventHandler;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.input.*;
import javafx.scene.layout.*;
import javafx.scene.text.Font;
import javafx.stage.Stage;

import java.util.*;

/**
 * GUI for application
 */
public class BoggleView {
    BoggleModel model; // The game for view to display

    Stage stage;

    Button newButton, exitButton, submit, next, credit, switchWords;// New game/ exit/ submit user input/ next game stage/ credit/ Switch between word lists

    Label p1score, p2score, sizeLabel;// the scores of two players/ Current size setting

    BorderPane borderPane;
    ScrollPane windowScrolling;
    public GridPane gridpane;
    GridPane creditPane;
    GridPane mainPane;

    TextField input, minutes, seconds, p1, p2;// for user input / minutes seconds of the timer/ two player names

    Slider size; // Slider to set the grid size

    public ChoiceBox<String> colorBlind;

    public ChoiceBox<Object> difficulties;
    ChoiceBox<Object> gameMode;

    ListView<String> curWords, bacWords;// Allow players to view their words

    HashMap<String, KeyCombination> hotkeys;

    boolean inSummary, inCredit;// If the GUI is in the summary page/ If the GUI is in the credit page

    public ArrayList<String> colorList = new ArrayList<>(Arrays.asList("white", "blue", "cyan", "green",
            "magenta", "orange", "red", "yellow", "pink"));

    final String forCredit = "Made by: Zhixing, Marvin, Matthew, and Tianran";

    final Label guide = new Label("Instruction:\nThe Boggle board contains a grid of letters that are randomly placed.\n" +
            "Both player going to try to find words in this grid by joining the letters.\n" +
            "You can form a word by connecting adjoining letters on the grid. Two\n" +
            "letters adjoin if they are next to each other horizontally, vertically, or\n" +
            "diagonally. The words you find must be at least 3 letters long, and you\n" +
            "can't use a letter twice in any single word. Your points will be based on\n" +
            "word length: a 3-letter word is worth 1 point, 4-letter words earn 2\n" +
            "points, and so on.\n\n" +
            "Keyboard only guide (Use below hotkeys with a shortcut key):\n" +
            "Players' names: up/down   |   (PVP/PVE/PVE: HOLE): P   |   Difficulties: D\n" +
            "Colorblind: C   |   Timer: T   |   Word list: L\n\n" +
            "(Timer only takes 0-99 minutes and 0-59 seconds. if the inputs are 0 or\n" +
            "non-natural numbers, no time limit for the round. Click on the timer\n" +
            "or press Any shortcut + SPACE to pause the timer during game.)\n\n" +

            "(Please use up and down arrow keys to navigate on the list, BACKSPACE\n" +
            "or double right click to delete the picked one)\n" +
            "Other hotkeys are specified on individual buttons.");

    final String[] colorModes = {"Red green blindness", "Red blindness", "Green blindness"};
    final String[] levels = {"Easy", "Normal", "Hard"};
    final String[] modes = {"PVP", "PVE", "PVE: HOLE"};


    public BoggleView(Stage stage) {
        //this.model = model;
        this.stage = stage;
        initUI();
    }

    /**
     * Initialize all GUI
     */
    private void initUI() {

        stage.setTitle("Boggle");

        borderPane = new BorderPane();
        borderPane.setStyle("-fx-background-color: #121212;");

        windowScrolling = new ScrollPane();
        windowScrolling.setFitToWidth(true);
        windowScrolling.setFitToHeight(true); // Readjust the window as soon as the contents within, change!
        windowScrolling.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);

        mainPane = new GridPane();
        mainPane.setPrefSize(700, 700);
        mainPane.setAlignment(Pos.CENTER);
        guide.setFont(new Font(20));
        guide.setStyle("-fx-text-fill: white");
        GridPane.setConstraints(guide, 0, 0);
        mainPane.getChildren().add(guide);

        gridpane = new GridPane();
        gridpane.setPrefSize(700, 700);
        gridpane.setAlignment(Pos.CENTER);

        // Names of two players
        p1 = new TextField();
        p1.setPromptText("P1's name");
        p1.setMaxWidth(140);
        p1.setFont(new Font(20));
        p1score = new Label("0");
        p1score.setPrefSize(140, 50);
        p1score.setFont(new Font(18));
        p1score.setStyle("-fx-text-fill: white");

        p2 = new TextField();
        p2.setPromptText("P2's name");
        p2.setMaxWidth(140);
        p2.setFont(new Font(20));
        p2score = new Label("0");
        p2score.setPrefSize(140, 50);
        p2score.setFont(new Font(18));
        p2score.setStyle("-fx-text-fill: white");


        VBox players = new VBox(10, p1, p1score, p2, p2score); //Scores below players' name
        players.setAlignment(Pos.TOP_CENTER);
        players.setPrefSize(150, 300);

        // Buttons on the left
        newButton = new Button("New Game (N)");
        newButton.setPrefSize(140, 50);
        newButton.setFont(new Font(16));
        newButton.setStyle("-fx-background-color: #17871b; -fx-text-fill: white;");

        // Create a size Label to help the user see the current value of the slider
        sizeLabel = new Label("Size: 5");
        sizeLabel.setFont(new Font(15));
        sizeLabel.setStyle("-fx-text-fill: white");

        // Create a difficulties Label to help the user see what the choice box below it does.
        Label diffLabel = new Label("Difficulty");
        diffLabel.setFont(new Font(13));
        diffLabel.setStyle("-fx-text-fill: white");

        // Create a Game Mode label to help the user see what the choice box below it does.
        Label gModeLabel = new Label("Game Mode");
        gModeLabel.setFont(new Font(13));
        gModeLabel.setStyle("-fx-text-fill: white");

        // Create a slider node and set it's color, fonts
        size = new Slider(4, 10, 5);
        size.setShowTickLabels(true);
        size.setBlockIncrement(1);
        VBox sizeSection = new VBox(sizeLabel, size);
        sizeSection.setAlignment(Pos.TOP_CENTER);
        sizeSection.setSpacing(3);

        colorBlind = new ChoiceBox<>();
        colorBlind.getItems().addAll(colorModes);
        colorBlind.setPrefSize(90, 30);
        colorBlind.setValue("Colorblind");

        difficulties = new ChoiceBox<>();
        difficulties.getItems().addAll(levels);
        difficulties.setPrefSize(90, 30);
        difficulties.setValue("Normal");
        VBox difficultySection = new VBox(diffLabel, difficulties);
        difficultySection.setAlignment(Pos.TOP_CENTER);
        difficultySection.setSpacing(3);

        gameMode = new ChoiceBox<>();
        gameMode.getItems().addAll(modes);
        gameMode.setPrefSize(90, 30);
        gameMode.setValue("PVE");
        VBox gameModeSection = new VBox(gModeLabel, gameMode);
        gameModeSection.setAlignment(Pos.TOP_CENTER);
        gameModeSection.setSpacing(3);

        /*
        add choice to choice box
         */
        colorBlind.setOnAction(this::set);

        Label colorLabel = new Label("Accessibility Options");
        colorLabel.setFont(new Font(13));
        colorLabel.setStyle("-fx-text-fill: white");

        VBox accessibilitySection = new VBox(colorLabel, colorBlind);
        accessibilitySection.setAlignment(Pos.TOP_CENTER);
        accessibilitySection.setSpacing(3);

        exitButton = new Button("Exit (ESC)");
        exitButton.setPrefSize(100, 25);
        exitButton.setFont(new Font(14));
        exitButton.setStyle("-fx-background-color: #17871b; -fx-text-fill: white;");

        VBox options = new VBox(15, newButton, sizeSection, gameModeSection, difficultySection, accessibilitySection, exitButton);
        options.setAlignment(Pos.TOP_CENTER);
        options.setPrefWidth(150);

        VBox left = new VBox(players, options); //Left of the borderpane

        credit = new Button("Credits"); // For credit page


        //Timer components
        minutes = new TextField();
        minutes.setPromptText("0 - 99");
        minutes.setPrefWidth(60);
        minutes.setText("0");
        Label mText = new Label("minutes");
        mText.setFont(new Font(15));
        mText.setStyle("-fx-text-fill: white");
        HBox min = new HBox(10, minutes, mText);
        min.setAlignment(Pos.CENTER);
        seconds = new TextField();
        seconds.setPromptText("0 - 59");
        seconds.setPrefWidth(60);
        seconds.setText("0");
        Label sText = new Label("seconds");
        sText.setFont(new Font(15));
        sText.setStyle("-fx-text-fill: white");
        HBox sec = new HBox(10, seconds, sText);
        sec.setAlignment(Pos.CENTER);
        VBox timer = new VBox(15, min, sec);
        VBox.setMargin(timer, new Insets(20, 0, 0, 20));
        timer.setPrefSize(130, 150 - credit.getHeight());

        VBox leftDown = new VBox(timer, credit);


        // Middle bottom part for player control
        input = new TextField();
        input.setPromptText("Press any shortcut + I to enter");
        input.setFont(new Font(34));
        input.setMaxWidth(600);

        submit = new Button("submit");
        submit.setPrefSize(80, 40);
        submit.setFont(new Font(15));
        submit.setStyle("-fx-background-color: #17871b; -fx-text-fill: white;");
        submit.setDisable(true);// Doesn't make sense to input a word if the game hasn't started

        next = new Button("next (G)");
        next.setPrefSize(80, 40);
        next.setFont(new Font(15));
        next.setStyle("-fx-background-color: #17871b; -fx-text-fill: white;");
        next.setDisable(true);// Same as the input above

        HBox controlButtons = new HBox(350, next, submit);
        controlButtons.setPrefWidth(700);
        controlButtons.setAlignment(Pos.CENTER);
        VBox control = new VBox(15, input, controlButtons);
        control.setAlignment(Pos.CENTER);

        switchWords = new Button("Switch (shortcut+W)"); // For user to see each other's word list in the summary page
        switchWords.setPrefWidth(150);
        switchWords.setDisable(true); // Disabled before the game
        HBox rightDown = new HBox(switchWords);
        rightDown.setPrefWidth(150);


        HBox bottom = new HBox(leftDown, control, rightDown); // Bottom of the borderpane
        bottom.setPrefHeight(150);

        Label title = new Label("Boggle");
        title.setFont(new Font(80));
        title.setStyle("-fx-text-fill: white");
        VBox top = new VBox(title);
        top.setPrefSize(1000, 150);
        top.setAlignment(Pos.CENTER);


        curWords = new ListView<>();
        curWords.setPrefWidth(150);
        bacWords = new ListView<>();
        bacWords.setPrefWidth(150);

        borderPane.setTop(top);
        borderPane.setLeft(left);
        borderPane.setRight(curWords);
        borderPane.setCenter(mainPane);
        borderPane.setBottom(bottom);

        initKeyCombo();

        windowScrolling.setContent(borderPane);

        var scene = new Scene(windowScrolling, 1000, 1000);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();

        newButton.setOnAction(e -> { // Start a new game
            inSummary = false;
            p1score.setText(String.valueOf(0));
            p2score.setText(String.valueOf(0));
            int gridSize = (int) size.getValue();
            String diff = (String) difficulties.getValue();

            GameBuilder builder;

            if (gameMode.getValue() == "PVP") {
                builder = new PVPBuilder();
            } else {
                p2.setText("Computer");
                p2.setEditable(false);
                if (gameMode.getValue() == "PVE: HOLE") {
                    builder = new HOLEBuilder();
                } else {
                    builder = new PVEBuilder();
                }
                builder.setDifficulty(diff);
            }
            builder.setSize(gridSize);

            initWords();
            if (model != null) {
                if (!model.game.pauseResume()) {
                    model.game.pauseResume();
                }
                else {
                    minutes.setText("");
                    seconds.setText("");
                }
            }
            model = builder.createGame();

            borderPane.setCenter(gridpane);
            roundUpdate();
            next.setDisable(false);
            borderPane.requestFocus();
        });

        size.setOnMouseDragged(e -> sizeLabel.setText("Size: " + (int) size.getValue())); // grid size
        size.setOnMouseClicked(e -> sizeLabel.setText("Size: " + (int) size.getValue()));
        size.setOnKeyReleased(e -> {
            KeyCode k = e.getCode();
            if (k == KeyCode.LEFT || k == KeyCode.RIGHT) {
                sizeLabel.setText("Size: " + (int) size.getValue());
            }
            else if (k == KeyCode.ENTER) borderPane.requestFocus();
        });

        gameMode.setOnAction(e -> { // Only for checking if the difficulties apply
            if (gameMode.getValue() == "PVP") { // No difficulty under PVP
                difficulties.setValue("None");
                difficulties.setDisable(true);
            }
            else {
                difficulties.setValue("Normal");
                difficulties.setDisable(false);
            }
            borderPane.requestFocus();
        });

        exitButton.setOnAction(e -> System.exit(0));

        credit.setOnAction(e -> { // Open the credit page
            if (creditPane == null) { // Lazy initialization
                creditPane = new GridPane();
                Label creditLabel = new Label(forCredit);
                creditLabel.setFont(new Font(20));
                creditLabel.setStyle("-fx-text-fill: white");
                GridPane.setConstraints(creditLabel, 0, 0);
                creditPane.getChildren().add(creditLabel);
                creditPane.setAlignment(Pos.CENTER);
            }
            if (inCredit) {
                newButton.setDisable(false); // Move back to the game page
                if (inSummary) borderPane.setCenter(gridpane);
                else borderPane.setCenter(mainPane);
            }
            else {
                if (!(inSummary || next.isDisable() || model == null)) {
                    pause();
                }
                newButton.setDisable(true); // Can't start a new game in credit page
                borderPane.setCenter(creditPane);
            }
            inCredit = !inCredit;
            borderPane.requestFocus();
        });

        p1.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) borderPane.requestFocus();
            else if (e.getCode() == KeyCode.DOWN) p2.requestFocus();
        });

        p2.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) borderPane.requestFocus();
            else if (e.getCode() == KeyCode.UP) p1.requestFocus();
        });

        input.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                //if (!submit.isDisable())
                    submit.fire();
            }
        });

        submit.setOnAction(e -> { // Submit a word for the current player
            if (model == null) return;
            String word = input.getText().strip();
            if (model.inputWord(word)) {
                ObservableList<String> l = curWords.getItems();
                if (!l.contains(word)) l.add(word);
            }
            input.setText("");
        });

        next.setOnAction(e -> { // Behaviors of the button next
            if (!inSummary) { // Not in the summary page
                if (model.next()) {
                    // If in the last turn of a round, go to the summary page
                    // To do: Need update the computer's word list in PVE mode (Waiting for difficulties)
                        pause();
                        inSummary = true;
                        gridpane.getChildren().clear(); // Clear the grid pane
                        // Set up the summary information
                        HashMap<Integer, String> summery = model.game.summarizeRound();
                        if (Objects.equals(p1.getText(), "")) {
                            p1.setText("Player 1");
                            curWords.getItems().set(0, "Player 1");
                        }
                        if (Objects.equals(p2.getText(), "")) {
                            p2.setText("Player 2");
                            bacWords.getItems().set(0, "Player 2");
                        }
                        Label sumInfo = new Label("It's been " + summery.get(0) + " round\n\n" + p1.getText() + ":\n" +
                                summery.get(1) + "\n\n" + p2.getText() + ":\n" + summery.get(2) + "\n\nClick \"next\" to " +
                                "continue or \"New Game\" to start a new game");
                        sumInfo.setFont(new Font(20));
                        sumInfo.setStyle("-fx-text-fill: white");
                        GridPane.setConstraints(sumInfo, 0, 0);
                        gridpane.getChildren().add(sumInfo);
                        // Update the total score under players' name
                        p1score.setText(String.valueOf(summery.get(1).split(": ")[2].split("\n")[0]));
                        p2score.setText(String.valueOf(summery.get(2).split(": ")[2].split("\n")[0]));
                        if (gameMode.getValue() == "PVE" || gameMode.getValue() == "PVE: HOLE") difficulties.setDisable(false);
                        gameMode.setDisable(false);
                        p1.setEditable(true);
                        p2.setEditable(true);
                        minutes.setEditable(true);
                        seconds.setEditable(true);
                        next.setDisable(false);
                        if (gameMode.getValue() == "PVE" || gameMode.getValue() == "PVE: HOLE") bacWords.getItems().addAll(model.game.getComputerWords());
                        switchWords.setDisable(false);
                } else { // Not in the last turn, go to next player's turn
                    model.game.renewTimer(minutes, seconds, input);
                    pause();
                    borderPane.setCenter(mainPane);
                    switchWords.setDisable(false);
                    switchWords.fire(); // switch to the next player's word list
                    switchWords.setDisable(true);
                }
                input.setDisable(false); // Resume the input field possibly disabled by the timer
            }
            else { // In the summary page, go to a new round
                int boardSize = (int) size.getValue();
                if (model.gameMode == "PVE") {
                    gameMode.setValue("PVE");
                } else if (model.gameMode == "PVE: HOLE") {
                    gameMode.setValue("PVE: HOLE");
                } else gameMode.setValue("PVP");
                initWords();
                roundUpdate();
                model.nextRound(boardSize);
                difficulties.setValue(model.getLevel());
                inSummary = false;
            }
            borderPane.requestFocus();
        });

        minutes.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) borderPane.requestFocus();
            else if (e.getCode() == KeyCode.DOWN) seconds.requestFocus();
        });

        seconds.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) borderPane.requestFocus();
            else if (e.getCode() == KeyCode.UP) minutes.requestFocus();
        });

        timer.setOnMouseClicked(e -> { // The game should be paused/resumed when clicking on the timer
            if (inSummary || inCredit || model == null) return; // No need to pause when in summary/credit page or if the game has not started.
            if (borderPane.getCenter() == gridpane) borderPane.setCenter(mainPane);
            pause();
            borderPane.requestFocus();
        });
        sText.setOnMouseClicked(timer.getOnMouseClicked());
        sec.setOnMouseClicked(timer.getOnMouseClicked());
        min.setOnMouseClicked(timer.getOnMouseClicked());
        mText.setOnMouseClicked(timer.getOnMouseClicked());


        // Set up the delete method for wor lists
        curWords.setOnKeyReleased(this::deleteWord);
        curWords.setOnMouseReleased(this::deleteWord);
        bacWords.setOnKeyReleased(this::deleteWord);
        bacWords.setOnMouseReleased(this::deleteWord);

        switchWords.setOnAction(e ->{
            ListView<String> temp = curWords; // Switch to the other word list
            curWords = bacWords;
            bacWords = temp;
            borderPane.setRight(curWords);
            borderPane.requestFocus();
        });

        borderPane.setOnKeyReleased(e -> { // Respond to hotkeys
            String k = "";
            for (String signifier : hotkeys.keySet()) {
                if (hotkeys.get(signifier).match(e)) {
                    k = signifier;
                    break;
                }
            }
            switch (k) {
                case "": break; // Not a hotkey
                case " ": {
                    if (inSummary || inCredit || model == null) return;
                    if (borderPane.getCenter() == gridpane) borderPane.setCenter(mainPane);
                    pause();
                    borderPane.requestFocus();
                    break;
                }
                case "N": { // new game
                    if (!newButton.isDisable()) newButton.fire();
                    borderPane.requestFocus();
                    break;
                }
                case "S": { // size slide
                    size.requestFocus();
                    break;
                }
                case "P": { // game mode
                    if (!gameMode.isDisable()) {
                        if (gameMode.getValue() == "PVP") {
                            gameMode.setValue("PVE");
                        } else if (gameMode.getValue().equals("PVE")) {
                            gameMode.setValue("PVE: HOLE");
                        } else gameMode.setValue("PVP");
                    }
                    borderPane.requestFocus();
                    break;
                }
                case "D": { // difficulties
                    if (!difficulties.isDisable()) {
                        String d = (String) difficulties.getValue();
                        switch (d) {
                            case "Normal" -> difficulties.setValue("Hard");
                            case "Hard" -> difficulties.setValue("Easy");
                            default -> difficulties.setValue("Normal");
                        }
                    }
                    borderPane.requestFocus();
                    break;
                }
                case "C": { // Open the credit page
                    if (!colorBlind.isDisable()) {
                        if (Objects.equals(colorBlind.getValue(), "Red green blindness")) {
                            colorBlind.setValue("Red blindness");
                        } else if (colorBlind.getValue().equals("Red blindness")) {
                            colorBlind.setValue("Green blindness");
                        } else colorBlind.setValue("Red green blindness");
                    }
                    colorBlind.fireEvent(new ActionEvent());
                    borderPane.requestFocus();
                    break;
                }
                case "E": { // Exit if on the board pane, go to the board pane otherwise
                    if (borderPane.isFocused()) System.exit(0);
                    else borderPane.requestFocus();
                }
                case "I": { // Go to input
                    input.requestFocus();
                    break;
                }
                case "G": { // next
                    if (!next.isDisable()) {
                        next.fire();
                        borderPane.requestFocus();
                    }
                    break;
                }
                case "p1": { // Go to p1's name field
                    p1.requestFocus();
                    break;
                }
                case "p2": { // p2's name
                    p2.requestFocus();
                    break;
                }
                case "T": { // Timer
                    if (!minutes.isDisable()) {
                        minutes.requestFocus();
                        break;
                    }
                }
                case "L": { // word list
                    curWords.requestFocus();
                }
            }
        });
        borderPane.requestFocus();
    }

    /**
     * Disable/enable nodes for a new round, assign the grid to the gridPane, also initialize the timer
     * Game modes, difficulties, the timer, players' name, and the list switcher are disabled when a round begins; submit is enabled
     */
    private void roundUpdate() {
        gameMode.setDisable(true);
        difficulties.setDisable(true);
        minutes.setEditable(false);
        seconds.setEditable(false);
        p1.setEditable(false);
        p2.setEditable(false);
        switchWords.setDisable(true);
        submit.setDisable(false);
        assign_letter_to_grid();
        model.game.renewTimer(minutes, seconds, input);
    }

    private void set(ActionEvent e) {
        new BoggleEventHandler(this).handle(e);
    }

    /**
     * Add all letter of the game grid to gridPane for display
     */
    protected void assign_letter_to_grid() {
        gridpane.getChildren().clear();
        char[][] grid = model.game.getGrid();
        int multiplier = 525 / grid.length; // Multiplier for letter font auto-adjusting
        int s = grid.length; // Grid size
        ArrayList<Label> children = new ArrayList<>(); // All Letters
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                Label n = new Label();
                n.setAlignment(Pos.TOP_CENTER);
                n.setStyle("-fx-text-fill: white");
                if (j < s - 1) {
                    n.setText(grid[j][i] + " "); //Add a space to the right of all letters, except those at the end of the col
                }
                else {
                    n.setText(Character.toString(grid[j][i]));
                }
                n.setFont(new Font(multiplier));
                children.add(n);
                GridPane.setConstraints(n, j, i);
            }
        }
        gridpane.getChildren().addAll(children);
    }

    /** Initialize hotkey combinations for GUI components
     *
     */
    private void initKeyCombo() {
        hotkeys = new HashMap<>();
        hotkeys.put("N", new KeyCodeCombination(KeyCode.N, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("S", new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("P", new KeyCodeCombination(KeyCode.P, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("D", new KeyCodeCombination(KeyCode.D, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("C", new KeyCodeCombination(KeyCode.C, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("E", new KeyCodeCombination(KeyCode.ESCAPE));
        hotkeys.put("I", new KeyCodeCombination(KeyCode.I, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("G", new KeyCodeCombination(KeyCode.G, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put(" ", new KeyCodeCombination(KeyCode.SPACE, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("p1", new KeyCodeCombination(KeyCode.UP, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("p2", new KeyCodeCombination(KeyCode.DOWN, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("T", new KeyCodeCombination(KeyCode.T, KeyCombination.SHORTCUT_DOWN));
        hotkeys.put("L", new KeyCodeCombination(KeyCode.L, KeyCombination.SHORTCUT_DOWN));
    }

    /**
     * Pause or resume the game
     */
    void pause() {
        if (model.game.pauseResume()) { // Pause
            submit.setDisable(true);
            next.setDisable(true);
        }
        else { // Resume
            submit.setDisable(false);
            borderPane.setCenter(gridpane);
            next.setDisable(false);
        }
    }

    /**
     * Initialize the word lists for both players
     */
    private void initWords() {
        curWords.getItems().clear();
        bacWords.getItems().clear();
        curWords.getItems().add(p1.getText());
        bacWords.getItems().add(p2.getText());
    }

    /**
     * Handle backspace on wordlist, remove the picked word
     */
    private void deleteWord(KeyEvent e) {
        if (e.getCode() == KeyCode.BACK_SPACE) {
            // Remove only when the player is allowed and the selected one is not the first (player's name)
            if (!(submit.isDisable() || input.isDisable()) && curWords.getSelectionModel().getSelectedIndex() > 0) {
                String word = curWords.getSelectionModel().getSelectedItem();
                curWords.getItems().remove(word);
                model.removeWord(word);
            }
        }
        else if (e.getCode() == KeyCode.ENTER) borderPane.requestFocus();
    }

    /**
     * Handle double right click on wordlist, remove the picked word
     */
    private void deleteWord(MouseEvent e) {
        if (e.getButton().equals(MouseButton.SECONDARY) && e.getClickCount() == 2) {
            // Remove only when the player is allowed and the selected one is not the first (player's name)
            if (!(submit.isDisable() || input.isDisable()) && curWords.getSelectionModel().getSelectedIndex() > 0) {
                String word = curWords.getSelectionModel().getSelectedItem();
                curWords.getItems().remove(word);
                model.removeWord(word);
            }
        }
    }

}


