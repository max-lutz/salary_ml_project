FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY models ./models
COPY nltkdata ./nltkdata


COPY requirements_api.txt  .
RUN  pip3 install -r requirements_api.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY api_app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api_app.handler" ]