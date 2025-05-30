#include <cmath>
#include <cstdint>
#include <ros/ros.h>
#include <ros/package.h>
#include <raylib.h>
#include <raymath.h>
#include "tanksim/actuator.h"
#include "tanksim/sensor.h"
#include "tanksim/link.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <queue>
#include <algorithm>
//#include <torch/torch.h>
#include <iostream>


/*#define STATE_SIZE 100
#define ACTION_SIZE 4

// ==== Globals ====
torch::Tensor state_tensor;
bool data_ready = false;
sensor_msgs::Image camera_data;
geometry_msgs::Pose2D gps_data;
float health = 1.0;

// === Policy Network (struct-based, not class) ===
struct PolicyNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    PolicyNetImpl(int input_size, int hidden_size, int output_size){
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};
TORCH_MODULE(PolicyNet);

torch::nn::Sequential net = torch::nn::Sequential(
    torch::nn::Linear(STATE_SIZE, 128),
    torch::nn::ReLU(),
    torch::nn::Linear(128, 128),
    torch::nn::ReLU(),
    torch::nn::Linear(128, ACTION_SIZE),
    torch::nn::Tanh()
);

// === Build a simple flat state tensor ===
torch::Tensor build_state() {
    std::vector<float> state(STATE_SIZE, 0.0f);

    // Fill some example data:
    state[0] = gps_data.x;
    state[1] = gps_data.y;
    state[2] = gps_data.theta;
    state[3] = health;

    // TODO: process image data and encode enemy locations etc

    return torch::from_blob(state.data(), {1, STATE_SIZE}).clone();
}

// === Publish action ===
void publish_action(const torch::Tensor& action_tensor) {
    auto* a = action_tensor.data_ptr<float>();

    std_msgs::Float32MultiArray drive;
    drive.data.push_back(a[0]); // left throttle
    drive.data.push_back(a[1]); // right throttle
    drive_pub.publish(drive);

    std_msgs::Float32MultiArray turret;
    turret.data.push_back(a[2]); // turret turn
    turret.data.push_back(a[3] > 0.8f ? 1.0f : 0.0f); // fire
    turret_pub.publish(turret);
}*/

/// STATE AND SENSORS

enum e_ros_sensors {
	ROS_SENSOR_GPS,
    ROS_SENSOR_IMU,
    ROS_SENSOR_DIAG,
    ROS_SENSOR_LIDAR,
    ROS_SENSOR_CAMERA,
    ROS_SENSOR_TURRET,

	ROS_SENSOR_LEN
};

const char* ros_sensors[] = {
	"sensor/gps",
	"sensor/imu",
	"sensor/diag",
	"sensor/lidar",
	"sensor/camera",
	"sensor/turret"
};

enum e_ros_actuators {
	ROS_ACTUATOR_DRIVE,
	ROS_ACTUATOR_TURRET,

	ROS_ACTUATOR_LEN
};

const char* ros_actuators[] = {
	"actuator/drive",
	"actuator/twist"
};

// invisible heuristic state input, storing 'most threatening target' or whatever
typedef struct {

} istate_t;

// visible heuristic state input tensor for GOAP and RL
typedef struct {

} vstate_t;

// heuristic output tensor
typedef struct {

} weight_t;

typedef struct {
	vstate_t precondition;
	vstate_t postcondition;
	float scalar; // rl backed cost scalar
} action_t;

typedef struct {
	// individual tank agents
	// friendly info (ie health) updated by central planner
	// enemy info updated by group vision, hit tracking etc.
	size_t id;
	float health,turret_ammo;
	float damage_given;
	float damage_taken;

	Vector3 position;
	Vector3 vel_linear;
	Vector3 vel_angular;
	Quaternion orientation;
	Quaternion turret_orientation;

	uint8_t los, turret_ready;
} agent_t;

typedef struct {
	int w,h,scalar;
	float* heightfield;
} map_t;

// world state
#define LIDAR_BIN 90
typedef struct {
	map_t map; // updated by local observation and central planner

	agent_t self;
	float lidar[LIDAR_BIN];
	std::vector<agent_t> friendlies; // updated real time
	std::vector<agent_t> combatants; // updated by local observation and central planner
	std::vector<size_t> objectives;
} slam_t;

typedef struct {
    double kp,ki,kd;
    double perr,integral,set;
} controller_pid_t;

enum pid_types {
	CONTROLLER_PID_MAIN_LINEAR,
	CONTROLLER_PID_MAIN_ANGULAR,
	CONTROLLER_PID_AUX_ANGULAR_PITCH,
	CONTROLLER_PID_AUX_ANGULAR_YAW,

	CONTROLLER_PID_LEN
};

typedef struct {
	controller_pid_t pid[CONTROLLER_PID_LEN];
} controller_t;

typedef struct {
	map_t* map;
	size_t* parents;
	float* costs;
	std::vector<std::pair<float,size_t>> open;
	std::vector<Vector2> path;
	size_t cell_current, cell_target, cell_target_last, cell_last_valid, path_idx;
	size_t stuck_time, stuck;

	float flank_radius;
} mp_planner_t;

// constant controller state nested giga struct
typedef struct {
	ros::Subscriber mission;
	ros::Subscriber subscribers[ROS_SENSOR_LEN];
    ros::Publisher publishers[ROS_ACTUATOR_LEN];
    std::vector<float> sensors[ROS_SENSOR_LEN];

    float sim_speed;
    slam_t* slam; // local knowledge
    mp_planner_t* planner; // motion planning
    controller_t* controller;
} ros_t;

inline float util_ema(float a, float b, float alpha){
	return alpha*b+(1.0f-alpha)*(a);
}

#define TAU 2*M_PI
void ros_sensor_callback(size_t type, std::vector<float> data, ros_t* ros){
	slam_t* slam = ros->slam;
	agent_t* self = &slam->self;
	float dt = ros->sim_speed;

	// switch for sensors
	// lidar updates terrain map
	switch(type){
		case ROS_SENSOR_LIDAR:{
			//printf("lidar!\n");
			// 90 samples returning distance and relative quaternion angle ( 5 floats )
			// assuming everything valid of course
			// proof of concept lidar SLAM in milsim
			size_t n = 0, o = 0;
			float min_angle = FLT_MAX, max_angle = -FLT_MAX;
			for(size_t i = 0; i < data.size(); i += 5){
				float d = data[i];
				Quaternion q_sensor = (Quaternion){data[i+1],data[i+2],data[i+3],data[i+4]};
				Quaternion q_abs = QuaternionMultiply(self->orientation,q_sensor);
				Vector3 v_direction = Vector3RotateByQuaternion((Vector3){0,0,1},q_abs);
				Vector3 v_rel = Vector3RotateByQuaternion((Vector3){0, 0, 1}, q_sensor);
				Vector3 v_pos = Vector3Add(Vector3Add(self->position,(Vector3){0,2,0}),Vector3Scale(v_direction,d));

				// 1. Project to XZ plane
			    Vector3 dir_xz = {v_rel.x, 0, v_rel.z};
			    float len = Vector3Length(dir_xz);
			    if (len < 0.001f) continue; // Avoid degenerate straight-down rays

			    // 2. Compute azimuth
			    float angle = atan2f(dir_xz.x, dir_xz.z); // atan2(z,x) gives CW angle from +X
			    if (angle < 0) angle += TAU; // Normalize to [0, 2PI)
			    min_angle = fminf(min_angle, angle);
    			max_angle = fmaxf(max_angle, angle);

			    // 3. Bin index
			    int bin = (int)floor(angle / (TAU/LIDAR_BIN))%LIDAR_BIN;
			    slam->lidar[bin] = util_ema(d,slam->lidar[bin],.5);
			    //printf("angle %.1f deg -> bin %d, dist %.1f\n", angle * RAD2DEG, bin, d);

				int xi = (int)v_pos.x/slam->map.scalar + slam->map.w/2;
				int zi = (int)v_pos.z/slam->map.scalar + slam->map.h/2;
				if (xi < 0 || xi >= slam->map.w || zi < 0 || zi >= slam->map.h) continue;
				float* h = &slam->map.heightfield[(zi) * slam->map.w + (xi)];
				if(*h < 0){
					*h = v_pos.y; // first sample
					n++;
				}else{
					*h = std::max(*h,v_pos.y);//util_ema(*h,v_pos.y,.5); // exponential moving average
					/*if(*h == v_pos.y){
						// if lidar is higher, inflate surrounding terrain
						const int radius = 2;
						for (int dx = -radius; dx <= radius; dx++) {
						    for (int dz = -radius; dz <= radius; dz++) {
						        int xi_n = xi + dx;
						        int zi_n = zi + dz;
						        if (xi_n < 0 || xi_n >= slam->map.w || zi_n < 0 || zi_n >= slam->map.h) continue;

						        float* h2 = &slam->map.heightfield[zi_n * slam->map.w + xi_n];
						        if(*h2 < 0){
						            *h2 = v_pos.y;
						        }else{
						            float dist = sqrtf(dx * dx + dz * dz);
						            float decay = 1.0f / (1.0f + dist);
						            *h2 = util_ema(*h2, v_pos.y * decay,.5);
						        }
						    }
						}
					}*/
					o++;
				}
			}
			//printf("Scan angle range: %.2f to %.2f radians\n", min_angle, max_angle);
			//printf("lidar: %zu reads, %zu new cells, %zu updated cells\n",data.size()/5,n,o);
		} break;
		case ROS_SENSOR_IMU:{
			// orientation
			self->orientation = (Quaternion){data[0],data[1],data[2],data[3]};

			// angular velocity
			self->vel_angular = (Vector3){data[4],data[5],data[6]};

			// linear velocity and fractional position value by acceleration
			// /Vector3 v_velocity_local = (Vector3){data[7]/dt,data[8]/dt,data[9]/dt};
			if (dt <= 0.0001f || std::isnan(dt) || std::isinf(dt)) {
		        ROS_ERROR("Invalid dt = %f in IMU sensor callback. Skipping integration.", dt);
		        break;
		    }

		    Vector3 v_velocity_local = (Vector3){data[7]/dt,data[8]/dt,data[9]/dt};

		    if (!std::isfinite(v_velocity_local.x) || !std::isfinite(v_velocity_local.y) || !std::isfinite(v_velocity_local.z)) {
		        ROS_ERROR("IMU linear velocity is non-finite: (%f, %f, %f)", v_velocity_local.x, v_velocity_local.y, v_velocity_local.z);
		        break;
		    }
			self->vel_linear = Vector3RotateByQuaternion(v_velocity_local, self->orientation);
			self->position = Vector3Add(self->position,self->vel_linear);

			//printf("imu read: q: %f %f %f %f, av: %f %f %f, lv: %f %f %f\n",
			//	data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9]);
		} break;
		case ROS_SENSOR_GPS:{
			//Vector3 v_frac = (Vector3){fmodf(self->position.x,1),fmodf(self->position.y,1),fmodf(self->position.z,1)};
			Vector3 v_gps = (Vector3){data[0],data[1],data[2]};
			float error = Vector3Distance(self->position,v_gps);
			const float max_error = 2.;
			float alpha = std::clamp(error/max_error,0.1f,0.9f);
			if(error > max_error) self->position = v_gps; // hard reset
			else self->position = (Vector3){util_ema(self->position.x,v_gps.x,alpha),util_ema(self->position.y,v_gps.y,alpha),util_ema(self->position.z,v_gps.z,alpha)};
			//printf("gps: %f %f %f, up: %f %f %f\n",data[0],data[1],data[2],self->position.x,self->position.y,self->position.z);
		} break;
		case ROS_SENSOR_CAMERA:{
			// track enemies TODO
			slam->combatants.clear();
			for(size_t i = 0; i < data.size(); i += 3){
				Vector3 pos_relative = (Vector3){data[i],data[i+1],data[i+2]};
				agent_t agent = {0};
				agent.position = Vector3Add(self->position,pos_relative);
				slam->combatants.push_back(agent);
			}
			printf("enemies seen this frame: %zu\n",slam->combatants.size());

		} break;
		case ROS_SENSOR_DIAG:{
			// track health
			self->health = data[0];
		} break;
		case ROS_SENSOR_TURRET:{
			// track ammo, turret orientation and reload timing
			self->turret_orientation = (Quaternion){data[0],data[1],data[2],data[3]};
			self->turret_ammo = data[4];
			self->turret_ready = (uint8_t)data[5];
		} break;
	}
}


/// MOTION PLANNING

inline size_t mp_get_idx(int x, int z, map_t* map){
	if(x < 0 || x >= map->w || z < 0 || z >= map->h) return SIZE_MAX;
	return z*map->w+x;
}

inline Vector2 mp_get_vec(size_t idx, map_t* map){
	return (Vector2){(float)(idx%map->w),(float)(idx/map->h)};
}

inline size_t mp_remap_idx(Vector2 vec, map_t* map){ // world space coords to grid cell
	Vector2 v_shift = Vector2Scale((Vector2){(float)map->w/2,(float)map->h/2},map->scalar);
	Vector2 v_scaled = Vector2Scale(Vector2Add(vec,v_shift),1.0f/map->scalar);
	return mp_get_idx((int)v_scaled.x,(int)v_scaled.y,map);
}

inline Vector2 mp_remap_vec(size_t idx, map_t* map){ // grid cell to world space coords
	Vector2 vec = mp_get_vec(idx,map);
	Vector2 v_shift = Vector2Scale((Vector2){(float)map->w/2,(float)map->h/2},map->scalar);
	return Vector2Subtract(Vector2Scale(vec,map->scalar),v_shift);
}

void mp_path(mp_planner_t* planner, size_t start, size_t end) {
	planner->path.clear();
    size_t current = end;
    Vector2 v_shift = (Vector2){(float)planner->map->w,(float)planner->map->h};
    size_t count = 1;
    //printf("PATH:[");
    while(current != start){
    	// rescale vectors
    	Vector2 vec = mp_remap_vec(current,planner->map);
    	//printf("(%.0f,%.0f),",vec.x,vec.y);
        planner->path.push_back(vec);//Vector2Subtract(Vector2Scale(mp_remap_vec(current,planner->map),(float)planner->map->w),v_shift));
        current = planner->parents[current];
        count++;
    }
    //printf("]\n");
    // add start
    //planner->path.push_back(mp_remap_vec(start,planner->map));
    planner->path_idx = planner->path.size()-1;
    //printf("new path: len %zu\n",count);
}

inline void mp_heap_push(std::vector<std::pair<float,size_t>>& heap, std::pair<float,size_t> val) {
    heap.push_back(val);
    std::push_heap(heap.begin(), heap.end(), std::greater<>());
}

inline std::pair<float,size_t> mp_heap_pop(std::vector<std::pair<float,size_t>>& heap) {
    std::pop_heap(heap.begin(), heap.end(), std::greater<>());
    auto val = heap.back();
    heap.pop_back();
    return val;
}

inline float mp_astar_euclidean(size_t a, size_t b, map_t* map){
	Vector2 v_a = mp_get_vec(a,map);
	Vector2 v_b = mp_get_vec(b,map);
	return Vector2Distance(v_a,v_b);
}

void mp_astar(ros_t* ros, size_t sidx, size_t eidx){
	mp_planner_t* planner = ros->planner;
	slam_t* slam = ros->slam;
	agent_t* self = &slam->self;
	if(planner->parents == NULL) planner->parents = (size_t*)calloc(planner->map->w*planner->map->h,sizeof(size_t));
	if(planner->costs == NULL) planner->costs = (float*)calloc(planner->map->w*planner->map->h,sizeof(float));
	planner->open.clear();
	// /memset(planner->parents,0,planner->map->w*planner->map->h*sizeof(size_t));
	//memset(planner->costs,0,planner->map->w*planner->map->h*sizeof(float));
	for(size_t i = 0; i < planner->map->w*planner->map->h; i++){
		planner->parents[i] = SIZE_MAX;
		planner->costs[i] = INFINITY;
	}

	/*Vector2 v_shift = (Vector2){(float)planner->map->w/2,(float)planner->map->h/2};
	Vector2 v_start_scaled = Vector2Scale(Vector2Add(start,v_shift),1.0f/planner->map->scalar);
	Vector2 v_end_scaled = Vector2Scale(Vector2Add(end,v_shift),1.0f/planner->map->scalar);
	size_t sidx = util_get_idx((int)v_start_scaled.x,(int)v_start_scaled.y,planner->map);
	size_t eidx = util_get_idx((int)v_end_scaled.x,(int)v_end_scaled.y,planner->map);*/

	std::vector<size_t> avoid = {};
	if(slam->combatants.size() > 0){
		for(size_t i = 0; i < slam->combatants.size(); i++){
    		agent_t* agent = &slam->combatants[i];
    		Vector2 pos = (Vector2){agent->position.x,agent->position.z};
    		avoid.push_back(mp_remap_idx(pos,planner->map));
    	}
    }

	mp_heap_push(planner->open,{mp_astar_euclidean(sidx,eidx,planner->map),sidx});
	planner->parents[sidx] = sidx;
	planner->costs[sidx] = 0;
	size_t cell_min = sidx;
	float dist_min = -1;
	uint8_t path_found = 0;
	size_t count = 0;

	while(!planner->open.empty()){
		count++;

        auto [fscore, current] = mp_heap_pop(planner->open);
        if(current == eidx){
            mp_path(planner,sidx,eidx);
            path_found = 1;
            break;
        }

        // ignore if path to this node is already shorter
        //if(planner->costs[current]+mp_astar_euclidean(current,eidx,planner->map) <= fscore) continue;
        //planner->costs[current] = fscore;
        Vector2 v_current = mp_get_vec(current,planner->map);
        float y = planner->map->heightfield[current];
        float g_current = planner->costs[current];

        int cx = current%planner->map->w;
        int cy = current/planner->map->w;

        // 8 direction astar with euclidean distance
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = cx + dx;
                int ny = cy + dy;
                size_t idx = mp_get_idx(nx,ny,planner->map);
                if (idx == SIZE_MAX) continue; // idx not in heightfield
                float e = planner->map->heightfield[idx];
                float g_scalar = e-y;
                //if(e == 0) g_scalar = 1;
                if(/*e <= 0 || */g_scalar > 1.f){
                	//planner->costs[current] += g_scalar/4.;
                	continue; // grade too steep or unseen
                }else{
                	// inflate costmap to ignore near wall
                	for (int ddy = -1; ddy <= 1; ddy++) {
			            for (int ddx = -1; ddx <= 1; ddx++) {
			                if (ddx == 0 && ddy == 0) continue;
			                int tx = nx + ddx;
			                int ty = ny + ddy;

			                size_t tidx = mp_get_idx(tx,ty,planner->map);
			                float te = planner->map->heightfield[tidx];
			                float tg_scalar = te-e;
			                if(tg_scalar > 1.f || tidx == SIZE_MAX) g_scalar += tg_scalar;
			            }
			        }
                }

                // cost for going near enemies
                float g_flank = 0;
                if(avoid.size() > 0){ // avoid is list of visible enemy occupied cells
                	for(size_t i = 0; i < avoid.size(); i++){
                		float dist = mp_astar_euclidean(idx, avoid[i], planner->map);
					    if (dist < planner->flank_radius) g_flank += fmax(1.0f - (dist / (planner->flank_radius/planner->map->scalar)),0.f);
                	}
                }


                float g_score = ((dx == 0 || dy == 0)?1.f:1.414f)*(1+fabsf(g_scalar));
                float tentative_g = g_current + g_score + g_flank; // cost from current to neighbor
                if(planner->costs[idx] <= tentative_g) continue;
                planner->costs[idx] = tentative_g;
                float h = mp_astar_euclidean(idx,eidx,planner->map);
                float f = tentative_g+h;
                if(dist_min == -1 || h < dist_min){
                	cell_min = idx;
                	dist_min = h;
                }

                mp_heap_push(planner->open, {f, idx});
                planner->parents[idx] = current;
            }
        }
    }
    if(path_found == 0) mp_path(planner,sidx,cell_min);
    printf("evaluated %zu cells\n",count);
}

float util_wrap_angle(float angle) {
    while(angle > M_PI) angle -= TAU;
    while(angle < -M_PI) angle += TAU;
    return angle;
}


Vector2 mp_vfh_plus(Vector2 target_dir, float* lidar, float radius) {
    const float k_attract = 2.f;    // Goal attraction strength
    const float k_repel = 4.0f;      // Obstacle repulsion strength
    float obstacle_radius = 2*radius; // Distance where repulsion kicks in

    Vector2 resultant_force = Vector2Zero();

    // 1. Add attraction force to target
    resultant_force = Vector2Scale(Vector2Normalize(target_dir), k_attract);

    // 2. Add repulsive forces from obstacles
    /*for(int i = 0; i < LIDAR_BIN; i++) {
        float distance = lidar[i];
        if(distance < obstacle_radius) {
            // Calculate obstacle direction (opposite of lidar beam direction)
            float angle = i*(TAU/LIDAR_BIN);
            Vector2 obstacle_dir = {cosf(angle), sinf(angle)};
            
            // Inverse-square law repulsion (stronger when closer)
            float strength = k_repel * (1.0f - (distance/obstacle_radius));
            //float strength = k_repel / (distance * distance + 1e-3f);
            Vector2 repel_force = Vector2Scale(obstacle_dir, -strength);
            
            resultant_force = Vector2Add(resultant_force, repel_force);
        }
    }*/

    for (int i = 1; i < LIDAR_BIN - 1; i++) {
	    float distance = lidar[i];
	    if (distance < obstacle_radius) {
	        float angle = i * (TAU / LIDAR_BIN);

	        // Neighbor lidar values
	        float d_prev = lidar[(i - 1) % LIDAR_BIN];
	        float d_next = lidar[(i + 1) % LIDAR_BIN];

	        float angle_prev = (i - 1) * (TAU / LIDAR_BIN);
	        float angle_next = (i + 1) * (TAU / LIDAR_BIN);

	        // Local wall surface
	        Vector2 p_prev = { cosf(angle_prev) * d_prev, sinf(angle_prev) * d_prev };
	        Vector2 p_next = { cosf(angle_next) * d_next, sinf(angle_next) * d_next };

	        Vector2 tangent = Vector2Subtract(p_next, p_prev);
	        tangent = Vector2Normalize(tangent);

	        // Normal = perpendicular to tangent
	        Vector2 normal = { -tangent.y, tangent.x };
	        normal = Vector2Normalize(normal);

	        // Calculate repel strength (tapered by angle & distance)
	        float strength = k_repel * 0.5f * (1.0f - (distance / obstacle_radius));

	        // Force direction: blend normal and tangent (bias away from wall but slide along it)
	        float bias = 0.5f; // 0 = full slide, 1 = full repulsion
	        Vector2 repel_dir = Vector2Normalize(Vector2Lerp(tangent, normal, bias));

	        // Accumulate scaled force
	        resultant_force = Vector2Add(resultant_force, Vector2Scale(repel_dir, strength));
	    }
	}

    // 3. Add tangential forces for smoother navigation
    /*for(int i = 0; i < LIDAR_BIN; i++) {
        if(lidar[i] < radius * 2.0f) {
        	float obstacle_angle = i * (TAU / LIDAR_BIN);

	        // Tangent direction is perpendicular to obstacle vector
	        // Decide +90° or -90° offset based on whether obstacle is to the left or right of goal
	        float relative = util_wrap_angle(atan2f(target_dir.y,target_dir.x) - obstacle_angle);  // goal_angle is robot->goal direction
	        float tangent_offset = (relative >= 0.0f) ? -TAU / 8.0f : TAU / 8.0f;  // +90° or -90°

	        float tangent_angle = obstacle_angle + tangent_offset;
            // Create tangential force to "flow" around obstacles
            //float angle = i * (TAU/LIDAR_BIN) + (TAU/8.0f); // 45° offset
            float proximity = 1.0f - (lidar[i] / (radius * 2.0f));
            Vector2 tangent_dir = {cosf(tangent_angle), sinf(tangent_angle)};
            Vector2 tangent_force = Vector2Scale(tangent_dir, 0.2 * k_repel * proximity * fmax(cosf(relative),0.f));
            resultant_force = Vector2Add(resultant_force,tangent_force);
        }
    }*/

    return Vector2Normalize(resultant_force);
}

typedef struct {
	float offset;
	float min;
	float mean;
} vfh_data;

vfh_data mp_vfh(float* lidar, float desired_heading_rad, float min_dist) {
    float best_score = -FLT_MAX;
	int best_bin = -1;
	float min_distance = FLT_MAX;
	float avg_distance = 0;

	for (int i = 0; i < LIDAR_BIN; i++) {
	    float angle = i*(TAU/LIDAR_BIN);  // Bin center angle in radians
	    float range = lidar[i];

	    avg_distance += range/LIDAR_BIN;
	    if(range < min_distance) min_distance = range;
	    if(range < min_dist) continue;

	    // Score: closeness to desired heading
	    float angle_diff = util_wrap_angle(desired_heading_rad-angle);
	    float score = -fabsf(angle_diff);  // Prefer smaller deviation

	    if (score > best_score) {
	        best_score = score;
	        best_bin = i;
	    }
	}

	if (best_bin >= 0) {
	    return (vfh_data){
	    	util_wrap_angle(best_bin*(TAU/LIDAR_BIN)),
	    	min_distance,
	    	avg_distance
	    };
	}
	return (vfh_data){0,min_distance,avg_distance};
}

float mp_pid(controller_pid_t* pid, float error, float dt){
    pid->integral = std::clamp(pid->integral+error*dt,-1.,1.);
    double derivative = (error-pid->perr)/dt;
    double output = std::clamp(pid->kp*error + pid->ki*pid->integral + pid->kd*derivative,-1.,1.);
    pid->perr = error;
    return output;
}

float mp_blend(float astar_heading, float vfh_heading, float vfhp_heading, float bias){
    float w_astar = 0.4+0.5*(1.-bias);
    float w_vfhp = 0.5*bias;//bias;
    float w_vfh = 0.1;

    float blended_x = cosf(astar_heading) * w_astar + cosf(vfh_heading) * w_vfh + cosf(vfhp_heading) * w_vfhp;
    float blended_y = sinf(astar_heading) * w_astar + sinf(vfh_heading) * w_vfh + sinf(vfhp_heading) * w_vfhp;

    return atan2f(blended_y, blended_x);
}

int mp_los(Vector2 to, float* lidar) {
    float dist = Vector2Length(to);
    float angle = atan2f(to.y, to.x); // relative to tank's heading
    float fov = 0.1f; // ~10 degrees each way, adjust as needed

    for (int i = 0; i < LIDAR_BIN; i++) {
        float beam_angle = i * (TAU / LIDAR_BIN); // assuming LIDAR is -π to +π
        float delta = util_wrap_angle(beam_angle - angle);
        if (fabsf(delta) < fov) {
            if (lidar[i] < dist - 0.1f) return 0; // Obstacle blocks direct line
        }
    }
    return 1;
}

void mp_travel(ros_t* ros){
	controller_t* controller = ros->controller;
	mp_planner_t* planner = ros->planner;
	agent_t* self = &ros->slam->self;
	slam_t* slam = ros->slam;
	Vector2 v_pos = (Vector2){self->position.x,self->position.z};
	size_t sidx = mp_remap_idx(v_pos,planner->map);
	//planner->map->heightfield[sidx] = self->position.y;

	size_t target = planner->cell_target;
	if(sidx != planner->cell_current || planner->cell_target != planner->cell_target_last) mp_astar(ros,sidx,target);
	planner->cell_target_last = planner->cell_target;
	planner->cell_current = sidx;

	Vector2 v_pos_goal = mp_remap_vec(planner->cell_target,planner->map);
	Vector2 v_pos_path = v_pos_goal;
	float v_distance_goal = Vector2Distance(v_pos_goal,v_pos);
	Vector3 forward = Vector3RotateByQuaternion((Vector3){0,0,1}, self->orientation);

	// coordinate and polar heading of tank
	Vector2 world_heading = (Vector2){forward.x,forward.z};
	float current_heading = atan2f(world_heading.y, world_heading.x);

	if(planner->path_idx > 0 && planner->path.size() > 0){
		v_pos_path = planner->path[planner->path_idx];
		/*for (int i = planner->path_idx - 1; i >= 0; i--) {
		    Vector2 candidate = planner->path[i];
		    if (mp_los(Vector2Rotate(Vector2Subtract(candidate,v_pos),-current_heading), slam->lidar) == 1) {
		        planner->path_idx = i;
		        //break;
		    }
		}*/
		if(Vector2Distance(v_pos,v_pos_path) < 2*planner->map->scalar) planner->path_idx--;
	}

	float v_distance_long = Vector2Distance(v_pos,v_pos_goal);
	float v_distance_short = Vector2Distance(v_pos,v_pos_path);

	// pathfinding coordinate goal
	Vector2 world_goal_astar = Vector2Normalize(Vector2Subtract(v_pos_path,v_pos));
	Vector2 local_goal_long = Vector2Rotate(Vector2Normalize(Vector2Subtract(v_pos_goal,v_pos)),-current_heading);
	float min_dist = FLT_MAX;

	// local orientation directions (apply algorithms to these)
	Vector2 local_goal_astar = Vector2Rotate(world_goal_astar, -current_heading);
	Vector2 local_goal_vfhp = mp_vfh_plus(local_goal_astar, slam->lidar, 8.f);
	Vector2 world_goal_vfhp = Vector2Rotate(local_goal_vfhp, current_heading);

	vfh_data vfh = mp_vfh(slam->lidar,0,16.);

	// local orientation offsets (desired path is some combination of these)
	float desired_heading_astar = (atan2f(local_goal_astar.y,local_goal_astar.x));
	float desired_heading_vfhp = (atan2f(local_goal_vfhp.y, local_goal_vfhp.x));
	float desired_heading_vfh = (vfh.offset);

	float proximity_bias = std::max(0.f,1.f-(std::max(0.f,vfh.min-4.f)/16.f));
	float alignment_factor = fmaxf(0.f, 0.8+0.2*fabsf(cosf(desired_heading_vfhp)));
	float final_bias = proximity_bias * alignment_factor;
	//float blend_bias = std::max(0.f,1.f-(min_dist/16.f));
	float desired_heading_blended = mp_blend(desired_heading_astar,desired_heading_vfh,desired_heading_vfhp,final_bias);
	float angle_difference = util_wrap_angle(desired_heading_blended);

	float forward_dist = slam->lidar[0];
	float lv = Vector3Length(self->vel_linear)*60;
	if(lv < .5 && final_bias > .9 && planner->stuck == 0){
		planner->stuck_time++;
		if(planner->stuck_time > 60) planner->stuck = 1;
	}
	if(planner->stuck_time > 0 && (lv >= .5 || planner->stuck == 1)) planner->stuck_time--;
	if(planner->stuck_time == 0) planner->stuck = 0;

	// angular pid
	float differential = cos(angle_difference);//fmaxf(cos(angle_difference),.2);//expf(-fabsf(angle_difference));
	float steering = mp_pid(&controller->pid[CONTROLLER_PID_MAIN_ANGULAR],angle_difference,ros->sim_speed);
	float throttle = mp_pid(&controller->pid[CONTROLLER_PID_MAIN_LINEAR],forward_dist,ros->sim_speed)*differential;
	float brake = 0;// 1.-1./min_dist;
	if(differential < 0.) steering = -steering;
	/*if(planner->stuck == 1){
		steering = -steering;
		throttle = -1;
	}	*/

	printf("dtg %.1f dtc %.1f pathlen: %zu\n",v_distance_long,v_distance_short,planner->path_idx);
	printf("control: %.2f| %.2f- %.2f/\n",throttle,steering,differential);
	printf("min dist: %.1f bias: %.1f stuck: %zu (%zu, %f)\n",vfh.min,final_bias,planner->stuck,planner->stuck_time,lv);
	printf("ha %.1f hvf %.1f hvp %.1f  :: hb %.1f (ad %.1f)\n",
		desired_heading_astar,desired_heading_vfh,desired_heading_vfhp,desired_heading_blended,angle_difference);
	
	float left = std::clamp(throttle - steering, -1.f, 1.f);
	float right = std::clamp(throttle + steering, -1.f, 1.f);

	tanksim::actuator msg;
	msg.actuator.push_back(left);
	msg.actuator.push_back(right);

	msg.actuator.push_back(brake);
	msg.actuator.push_back(brake);
	ros->publishers[ROS_ACTUATOR_DRIVE].publish(msg);
}

void mp_aim(ros_t* ros){
	controller_t* controller = ros->controller;
	mp_planner_t* planner = ros->planner;
	agent_t* self = &ros->slam->self;
	slam_t* slam = ros->slam;

	agent_t* target = NULL;
	float target_cost = FLT_MAX;

	if(slam->combatants.size() > 0){
		for(size_t i = 0; i < slam->combatants.size(); i++){
			agent_t* candidate = &slam->combatants[i];
			float cost = Vector3Distance(self->position,candidate->position)+candidate->damage_given-candidate->damage_taken;
			if(cost < target_cost){
				target = candidate;
				target_cost = cost;
			}
		}
	}

	/*Vector3 forward = Vector3RotateByQuaternion((Vector3){0,0,1}, self->orientation);
	Vector3 forward_turret = Vector3RotateByQuaternion((Vector3){0,0,1}, QuaternionMultiply(self->orientation,self->turret_orientation));
	Vector3 euler_turret = QuaternionToEuler(self->turret_orientation);
	//Vector3 euler_forward = QuaternionToEuler(QuaternionMultiplyself->orientation);

	float yaw_error;

	if(target != NULL){

	}*/

	float turret_yaw_error = 0;
    float turret_pitch_error = 0;
    float fire = 0;

    Vector3 turret_euler = QuaternionToEuler(self->turret_orientation); // Assuming yaw = y, pitch = x

    float desired_yaw = 0;
    float desired_pitch = 0;
    float current_yaw = turret_euler.y;
    float current_pitch = turret_euler.x;

    if (target != NULL) {
        Vector3 to_target_world = Vector3Subtract(target->position, self->position);

        // Rotate into self's local frame
        Quaternion inv_body_orientation = QuaternionInvert(self->orientation);
        Vector3 to_target_local = Vector3RotateByQuaternion(to_target_world, inv_body_orientation);

        // --- 3. Desired yaw and pitch to point turret at target ---
        float desired_yaw = atan2f(to_target_local.x, to_target_local.z);
        float horizontal_dist = sqrtf(to_target_local.x * to_target_local.x + to_target_local.z * to_target_local.z);
        float desired_pitch = atan2f(to_target_local.y, horizontal_dist); // consider articulation limit

        // --- 4. Current turret orientation ---

        turret_yaw_error = util_wrap_angle(desired_yaw-current_yaw);
        turret_pitch_error = util_wrap_angle(desired_pitch-current_pitch);

        if(Vector2Length((Vector2){turret_yaw_error,turret_pitch_error}) < .11) fire = 1;

        //printf("desired: %.1f %.1f\n",desired_pitch,desired_yaw);
        //printf("euler: %.1f %.1f\n",turret_euler.x, turret_euler.y);
        printf("target visible: ye: %.1f pe: %.1f f: %.0f\n",turret_yaw_error,turret_pitch_error,fire);
    }else{
    	turret_yaw_error = util_wrap_angle(desired_yaw-current_yaw);
        turret_pitch_error = util_wrap_angle(desired_pitch-current_pitch);
    }

    float twist_yaw = mp_pid(&controller->pid[CONTROLLER_PID_AUX_ANGULAR_YAW],turret_yaw_error,ros->sim_speed);
	float twist_pitch = mp_pid(&controller->pid[CONTROLLER_PID_AUX_ANGULAR_PITCH],turret_pitch_error,ros->sim_speed);
	printf("output: yaw: %.1f, pitch %.1f\n",twist_yaw,twist_pitch);


	tanksim::actuator msg;
	msg.actuator.push_back(twist_yaw);
	msg.actuator.push_back(twist_pitch);
	msg.actuator.push_back(fire);
	/*msg.actuator.push_back(left);
	msg.actuator.push_back(right);

	msg.actuator.push_back(brake);
	msg.actuator.push_back(brake);*/
	ros->publishers[ROS_ACTUATOR_TURRET].publish(msg);
}

void mp_plan(ros_t* ros){
	slam_t* slam = ros->slam;
	agent_t* self = &ros->slam->self;

	size_t target_idx = SIZE_MAX;
	float min_dist = FLT_MAX;
	float combat_avg_dist = 0;
	if(slam->combatants.size() > 0){
		for(size_t i = 0; i < slam->combatants.size(); i++){
			agent_t* agent = &slam->combatants[i];
			float d = Vector3Distance(self->position,agent->position);
			combat_avg_dist += d/slam->combatants.size();
			if(d < min_dist){
				min_dist = d;
				target_idx = mp_remap_idx((Vector2){agent->position.x,agent->position.z},ros->planner->map);
				ros->planner->flank_radius = combat_avg_dist/4;
			}	
		}
	}
	ros->planner->flank_radius = combat_avg_dist;
	// go to nearest objective if its close than a visible enemy
	if(slam->objectives.size() > 0){
		for(size_t i = 0; i < slam->objectives.size(); i++){
			size_t idx = slam->objectives[i];
			Vector2 pos = mp_remap_vec(idx,ros->planner->map);
			float d = Vector2Distance((Vector2){self->position.x,self->position.z},pos);
			if(d < min_dist){
				min_dist = d;
				target_idx = idx;
			}
		}
	}

	if(target_idx == SIZE_MAX){
		// random somewhere in the middle if its already near its last target
		if(Vector2Distance((Vector2){self->position.x,self->position.z},mp_remap_vec(ros->planner->cell_target,ros->planner->map)) < 2){
			ros->planner->cell_target = mp_get_idx(
				ros->planner->map->w/8+rand()%ros->planner->map->w/4, 
				ros->planner->map->w/8+rand()%ros->planner->map->w/4, 
				ros->planner->map
			);
		}
	}else ros->planner->cell_target = target_idx;

	printf("combatants: %zu objectives: %zu avg_combatant: %.1f min_dist: %.1f target: %zu\n",
		slam->combatants.size(),slam->objectives.size(),combat_avg_dist,min_dist,target_idx);
}

void ros_mission_callback(std::vector<float> data, ros_t* rosptr){
	rosptr->slam->objectives.clear();
	for(size_t i = 0; i < data.size(); i += 2){
		Vector2 pos = (Vector2){data[i],data[i+1]};
		rosptr->slam->objectives.push_back(mp_remap_idx(pos,rosptr->planner->map));
	}
}


int main(int argc, char** argv){
	std::string node_name = "tank_controller_" + std::to_string(getpid());
    ros::init(argc, argv,node_name);
    ros::NodeHandle nh_anon;
    srand(time(NULL));

    ros::ServiceClient client = nh_anon.serviceClient<tanksim::link>("link");
	tanksim::link srv;
	std::string ns;
	if (!client.call(srv)) {
		printf("SIMULATOR NOT RUNNING\n");
		return 1;
	}
    uint32_t id = srv.response.id;
    float sim_speed = srv.response.speed;
    uint8_t team = srv.response.team;
    uint32_t map_width = srv.response.width;
    uint32_t map_height = srv.response.height;
    ns = "tank_" + std::to_string(id);
    printf("linked to namespace %s\n",ns.c_str());
    ros::NodeHandle nh(ns);
    // env init
    ros_t ros; 
    ros.sim_speed = sim_speed;
    slam_t slam = {0};
    slam.map.scalar = 4;
    slam.map.w = map_width/slam.map.scalar;
    slam.map.h = map_height/slam.map.scalar;
    slam.map.heightfield = (float*)calloc(slam.map.w*slam.map.h,sizeof(float));
    for(size_t i = 0; i < LIDAR_BIN; i++) slam.lidar[i] = FLT_MAX;

    mp_planner_t mp_planner = {0};
    mp_planner.map = &slam.map;
    mp_planner.cell_target = rand()%(mp_planner.map->w*mp_planner.map->h);

    controller_t controller = {0};
    controller.pid[CONTROLLER_PID_MAIN_ANGULAR] = (controller_pid_t){1,0.1,0.2, 0.,0.,2.};
    controller.pid[CONTROLLER_PID_MAIN_LINEAR] = (controller_pid_t){1.,.05,.2, 0.,0.,5.};
    controller.pid[CONTROLLER_PID_AUX_ANGULAR_PITCH] = (controller_pid_t){1,0.1,0.2, 0.,0.,2.};
    controller.pid[CONTROLLER_PID_AUX_ANGULAR_YAW] = (controller_pid_t){1.,0.1,0.2, 0.,0.,5.};


    ros.slam = &slam;
    ros.planner = &mp_planner;
    ros.controller = &controller;
    ros_t* rosptr = &ros;

    for(size_t i = 0; i < ROS_SENSOR_LEN; i++){
    	ros.subscribers[i] = nh.subscribe<tanksim::sensor>(
	        ros_sensors[i], 1,
	        [i, rosptr](const tanksim::sensor::ConstPtr& msg) { ros_sensor_callback(i, msg->data, rosptr); } // more evil pointer capture
	    );
	    printf("subscribed to %s\n",ros_sensors[i]);
    }

	for(size_t i = 0; i < ROS_ACTUATOR_LEN; i++){
		ros.publishers[i] = nh.advertise<tanksim::actuator>(ros_actuators[i],1);
		printf("advertising %s\n",ros_actuators[i]);
	}

	ros.mission = nh_anon.subscribe<tanksim::sensor>("mission",1,[rosptr](const tanksim::sensor::ConstPtr& msg) { ros_mission_callback(msg->data, rosptr); });



    ros::Rate loop((double)sim_speed);

    while (ros::ok()) {
        ros::spinOnce();
        mp_travel(rosptr);
        mp_aim(rosptr);
        mp_plan(rosptr);
        loop.sleep();
    }

    return 0;
}
