package filter;

import java.util.List;

/**
 * @Author Skyrimgo
 * @Contact ashassnow@126.com
 * @date 2018/10/10 16:19
 */
public interface ParticleFilter {

    List<Double> getResultValues(Double primaryValue, List<Double> observedValues);

}
