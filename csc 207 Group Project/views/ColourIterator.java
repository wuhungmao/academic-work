package views;

import javafx.scene.paint.Color;

import java.util.*;
import java.util.function.Consumer;

/**
 * An iterator over a list of colours.
 */
public class ColourIterator implements Iterator<Color>{
    private Color[] colours;
    private int index;

    /**
     * Constructor that uses a list to generate an iterator.
     * @param colours
     */
    public ColourIterator(List<Color> colours) {
        this.colours = colours.toArray(new Color[0]);
        this.index = 0;
    }

    /**
     * Check if the iterator contains a further element.
     * @return
     */
    @Override
    public boolean hasNext() {
        return (!Objects.isNull(this.colours[index])) && index < this.colours.length;
    }

    /**
     * Returns the next colour in the list.
     * @return
     */
    @Override
    public Color next() {
        return colours[index++];
    }

    /**
     * Return a new iterator according to a colour filter.
     * @param filter the filter ColourFilter that is applied.
     * @return
     */
    public ColourIterator filter(final ColourFilter filter) {
        List<Color> list = Arrays.asList(this.colours);
        List<Color> filtered = new ArrayList<>();
        for (Color color : list) {
            if (filter.filter(color)) {
                filtered.add(color);
            }
        }
        return new ColourIterator(filtered);
    }
}
