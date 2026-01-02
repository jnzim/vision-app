#pragma once
#include <deque>
#include <cmath>
#include <cstddef>

class RollingStats {
public:
    explicit RollingStats(std::size_t window) : m_window(window) {}

    void add(double x) 
    {
        if (m_window == 0) return;

        if (m_vals.size() == m_window) 
        {
            const double old = m_vals.front();
            m_vals.pop_front();
            m_sum   -= old;
            m_sumSq -= old * old;
        }
        m_vals.push_back(x);
        m_sum   += x;
        m_sumSq += x * x;
    }

    std::size_t count() const { return m_vals.size(); }

    double mean() const 
    {
        if (m_vals.empty()) 
        {    
            return 0.0;
        }
        return m_sum / static_cast<double>(m_vals.size());
    }

    double stddev() const 
    {
        const std::size_t n = m_vals.size();
        if (n < 2) 
        {
            return 0.0;
        }
        const double mu = mean();
        double v = (m_sumSq / static_cast<double>(n)) - (mu * mu);
        if (v < 0.0) 
        {
            v = 0.0;
        }
        return std::sqrt(v);
    }

private:
    std::size_t m_window;
    std::deque<double> m_vals;
    double m_sum = 0.0;
    double m_sumSq = 0.0;
};
