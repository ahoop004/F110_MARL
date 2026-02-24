from setuptools import setup

package_name = 'car_crashers'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        package_name: ['ppo_policy.pt'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ftg.launch.py', 'launch/ppo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agilex',
    maintainer_email='agilex@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ftg_node = car_crashers.ftg_node:main',
            'td3_node = car_crashers.TD3:main',
            'dqn_node = car_crashers.RainbowDQN:main',
            'sac_node = car_crashers.SAC:main',
            'ppo_node = car_crashers.PPO:main',
        ],
    },
)
