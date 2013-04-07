#!/usr/bin/env python

from random import *
from math import *
import numpy as np


def train_models(times_signals):
    """Train models by fitting Gaussian distribution parameters from signals

    :type times_signals: dict
    :param times_signals: dictionary contraining times and their corresponding signals;
                          key is time string formated as "hh:mm:ss";
                          value is a numpy.array contraining the signals for a particular time

    """

    # models is a dictionary contraining times and the fitted Gaussian parameter (mu, sigma);
    # key is time string formated as "hh:mm:ss";
    # value is the parameter tuple (mu, sigma)
    models = {}
    for time, signals in times_signals.items():
        # fit maximum likelihood parameters for Gaussian distribution
        models[time] = (signals.mean(), signals.std())

    return models


def norm_PDF(x, mu, sigma):
    """Calculate the Gaussian probability density function (PDF) at the input value,
    given the mean (mu) and standard deviation (sigma)

    :type x: double
    :param x: input value

    :type mu: double
    :param mu: mean of Gaussian distribution

    :type sigma: double
    :param sigma: standard deviation of Gaussian distribution

    """

    # PDF is calculated in log space and then coverted back to prevent numerical underflow
    return exp(-(x - mu)**2 / (2 * sigma**2) - log(sqrt(2 * pi) * sigma))


def uniform_PDF(x, lower=0, upper=1):
    """Calculate the probability density function of uniform distribution (PDF) at the input value

    """

    return 1.0 / (upper - lower)


def signal_cond_abnormal(signal, time):
    """Calculate probability of signal conditioned on the machine being abnormal

    the signal  of an abnormal machine is uniformly distributed in [0, 1]
    """

    return uniform_PDF(signal, 0, 1);


def signal_cond_normal(signal, time, models):
    """Calculate probability of signal conditioned on the machine being normal

    the distribution of signal of an normal machine is Gaussian distribution

    :type signal: double
    :param signal: signal at the given time, 0.0 <= signal <= 1.0

    :type time: string
    :param time: time string, format "hours:minutes:seconds"

    :type models: dict
    :param models: signal model of different times; a signal model is
                             Gaussian distribution parameterized by mu and sigma

    """

    return norm_PDF(signal, models[time][0], models[time][1])



def is_abnormal(times, signals, models, normal_prior=0.9, abnormal_prior=0.1):
    """
    Use Naive Bayes classifier to decide whether this machine is in
    abnormal state given the signals of a period time

    P(abnormal | signals) = P(abnormal, signals) / P(signals)
    P(signals) = P(signals, normal) + P(signals, abnormal)

    In a Naive Bayes model,
    the signal states are independent of each other given the machine status
    (i.e. features are independent given the label),
    we must have

    P(signals, normal) = P(normal) * P(signal1 | normal) * P(signal2 | normal) * ... * P(signalN | normal)
    P(signals, abnormal) = P(abnormal) * P(signal1 | abnormal) * P(signal2 | abnormal) * ... * P(signalN | abnormal)


    """

    # P(signals, normal)
    joint_signals_normal = normal_prior
    for signal, time in zip(signals, times):
        joint_signals_normal *= signal_cond_normal(signal, time, models)

    # P(signals, abnormal)
    joint_signals_abnormal = abnormal_prior
    for signal, time in zip(signals, times):
        joint_signals_abnormal *= signal_cond_abnormal(signal, time)

    return joint_signals_abnormal > joint_signals_normal

    # # P(signals) = P(signals, normal) + P(signals, abnormal)
    # joint_signals = joint_signals_normal  + joint_signals_abnormal

    # # P(abnormal | signals) = P(abnormal, signals) / P(signals)
    # abnormal_cond_signals = joint_signals_abnormal / joint_signals;

    # return abnormal_cond_signals > 0.5


def randtime():
    return format_time(randrange(0, 24), randrange(0, 60), randrange(0, 60))

def format_time(hour, minute, second):
    return str(hour).zfill(2) + ':' + str(minute).zfill(2) + ':' + str(second).zfill(2)

def test_perform():

    # synthesize cpu usage data
    mu = 0.5
    sigma = 0.12
    times_signals = {}
    for hour in range(0, 24):
        for minute in range(0, 60):
            for second in range(0, 60):
                times_signals[format_time(hour, minute, second)] = np.random.normal(mu, sigma, 50)


    # train cpu usages models
    models = train_models(times_signals)

    # test the accuracy
    ntest = 200
    duration = 20

    print """cpu usage data generated from the same Gaussian distribution as training data,
consider as normal"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.normal(mu, sigma, duration)
        correct += int(is_abnormal(times, signals, models) is False)
    print 'accuracy: ' + str(correct / float(ntest)) + '\n'


    print """cpu usage data generated from the same Gaussian distribution as training data,
average cpu usage incrase 0.15
consider as abnormal?"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.normal(mu + 0.15, sigma, duration)
        correct += int(is_abnormal(times, signals, models) is True)
    print 'accuracy: ' + str(correct / float(ntest)) + '\n'


    print """cpu usage data generated from the same Gaussian distribution as training data
cpu usage deviation increase 0.1
onsider as abnormal?"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.normal(mu, sigma + 0.1, duration)
        correct += int(is_abnormal(times, signals, models) is True)
    print 'accuracy: ' + str(correct / float(ntest)) + '\n'


    print """cpu usage data generated from the same Gaussian distribution as training data
average cpu usage incrase 0.15
cpu usage deviation increase 0.1
consider as abnormal"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.normal(mu + 0.15, sigma + 0.1, duration)
        correct += int(is_abnormal(times, signals, models) is True)
    print 'accuracy: ' + str(correct / float(ntest)) + '\n'


    print """cpu usage data generated from normal distribution
consider as abnormal"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.normal(0, 1, duration)
        correct += int(is_abnormal(times, signals, models) is True)
    print 'accuracy: ' + str(correct / float(ntest)) + '\n'


    print """cpu usage data generated from beta distribution
consider as abnormal"""
    correct = 0
    for i in range(ntest):
        times = [format_time(8, 10, second) for second in range(0, duration)]
        signals = np.random.beta(2, 5, duration)
        correct += int(is_abnormal(times, signals, models) is True)
    print 'accuracy: ' + str(correct / float(ntest))

if __name__ == "__main__":
    test_perform()
