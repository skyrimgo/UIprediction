package entity;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @Author Skyrimgo
 * @Contact ashassnow@126.com
 * @date 2018/10/10 16:19
 */
@Data
@NoArgsConstructor
public class Particle implements Comparable<Particle>, Cloneable{

    private double value;
    private double weight;

    public Particle(Particle particle){
        this.value = particle.getValue();
        this.weight = particle.getWeight();
    }

    @Override
    public int compareTo(Particle o) {
        double oValue = o.getValue();
        if(this.value == oValue){
            return 0;
        } else if(this.value > oValue){
            return 1;
        } else{
            return -1;
        }
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
