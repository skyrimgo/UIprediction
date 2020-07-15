package entity;

public class Data {
    /* Hibernate要求类中有一个属性值唯一，对应表中的主键 */
    private int d_id;
    private double d_data;

    /**
     * @param d_id the d_id to set
     */
    public void setD_id(int d_id) {
        this.d_id = d_id;
    }

    /**
     * @return the d_id
     */
    public int getD_id() {
        return d_id;
    }

    /**
     * @param d_data the d_data to set
     */
    public void setD_data(double d_data) {
        this.d_data = d_data;
    }

    /**
     * @return the d_data
     */
    public double getD_data() {
        return d_data;
    }

    @Override
    public String toString() {
        return "data:[d_id=" + d_id + "d_data=" + d_data + "]";
    }

}
